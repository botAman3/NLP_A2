import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import spacy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Modify the tokenizer to keep contractions together
infixes = nlp.Defaults.infixes
infixes = [x for x in infixes if x not in [r"\'"]]
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)

# Load pre-trained FastText embeddings
FASTTEXT_PATH = "/home/aasif057/nlp_embeddings/cc.en.300.vec"
fast_vectors = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False)

# Load GloVe embeddings
GLOVE_PATH = "/home/aasif057/nlp_embeddings/glove.840B.300d.txt"
def load_glove_embeddings(file_path):
    word_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype="float32")
                if len(vector) == 300:
                    word_vectors[word] = vector
            except ValueError:
                pass
    return word_vectors

glove_vectors = load_glove_embeddings(GLOVE_PATH)

# Combine FastText and GloVe
EMBEDDING_DIM = 300
def get_embedding(word):
    if word in fast_vectors and word in glove_vectors:
        return torch.tensor((fast_vectors[word] + glove_vectors[word]) / 2, dtype=torch.float32)
    elif word in fast_vectors:
        return torch.tensor(fast_vectors[word], dtype=torch.float32)
    elif word in glove_vectors:
        return torch.tensor(glove_vectors[word], dtype=torch.float32)
    else:
        return torch.zeros(EMBEDDING_DIM)

# Normalize token
def normalize_token(token):
    return token.replace("'", "")

# Find aspect index in tokenized sentence
def find_aspect_index(sentence_tokens, aspect_tokens):
    for i in range(len(sentence_tokens)):
        if sentence_tokens[i : i + len(aspect_tokens)] == aspect_tokens:
            return i
    return -1

# Preprocess data
def preprocess_data(json_data):
    preprocessed_data = []
    for data in json_data:
        sentence = data['sentence']
        aspect_terms = data.get('aspect_terms', [])
        doc = nlp(sentence)
        tokens = [normalize_token(token.text) for token in doc if not token.is_punct]
        for aspect in aspect_terms:
            aspect_term = aspect['term']
            polarity = aspect['polarity']
            aspect_doc = nlp(aspect_term)
            aspect_tokens = [normalize_token(token.text) for token in aspect_doc]
            index = find_aspect_index(tokens, aspect_tokens)
            if index != -1:
                preprocessed_data.append({
                    "tokens": tokens,
                    "polarity": polarity,
                    "aspect_term": aspect_term,
                    "index": index
                })
    return preprocessed_data
def save_preprocessed_data(file_path,data):
     save_path = file_path.replace(".json","") + "_task_2.json"
     with open(save_path, "w", encoding="utf-8") as file:
         json.dump(data, file, indent=4)

def load_and_preprocess(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    preprocessed_data = preprocess_data(json_data)
    save_preprocessed_data(file_path,preprocessed_data)
    return preprocessed_data

# Custom Dataset
class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"]
        aspect_term = sample["aspect_term"][0]
        index = sample["index"]
        polarity = sample["polarity"]

        # Get embeddings
        token_embeddings = torch.stack([get_embedding(tok) for tok in tokens])  # (seq_len, embed_dim)
        aspect_embedding = get_embedding(aspect_term)  # (embed_dim,)

        # Convert polarity to label
        
        polarity_map = {"positive": 2, "negative": 0, "neutral": 1, "conflict": 3}
        polarity_label = polarity_map[polarity]

        return token_embeddings, aspect_embedding, index, polarity_label

# Collate function for batching
def collate_fn(batch):
    token_indices, aspect_indices, indices, polarities = zip(*batch)

    # Convert token sequences to a single tensor with padding
    token_indices_padded = pad_sequence(token_indices, batch_first=True, padding_value=0.0)

    # Convert other items to tensors
    aspect_indices = torch.stack(aspect_indices)  # Stack aspect embeddings
    indices = torch.tensor(indices, dtype=torch.long)
    polarities = torch.tensor(polarities, dtype=torch.long)

    # Compute sequence lengths
    lengths = torch.tensor([len(seq) for seq in token_indices], dtype=torch.int64)

    return token_indices_padded, aspect_indices, indices, polarities, lengths

def prepare_dataset(train_path, val_path):
    train_data = load_and_preprocess(train_path)
    val_data = load_and_preprocess(val_path)
    train_loader = DataLoader(ABSADataset(train_data), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ABSADataset(val_data), batch_size=16, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

# Define TD_LSTM_Attention model

class TD_LSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(TD_LSTM_Attention, self).__init__()
        self.left_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.right_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.aspect_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 6, num_classes)
        self.dropout = nn.Dropout(0.4)
    def forward(self, x, aspect, index):
        # left_context, right_context = torch.zeros_like(x), torch.zeros_like(x)
        # for i in range(x.shape[0]):
        #     left_context[i, :index[i]] = x[i, :index[i]]
        #     right_context[i, -len(x[i, index[i]+1:]):] = x[i, index[i]+1:]
        # left_context = torch.zeros_like(x)
        # right_context = torch.zeros_like(x)
        left_contexts, right_contexts = [], []
        for i in range(x.shape[0]):  # Iterate over batch
            left_contexts.append(x[i, :index[i]])  # Left context
            
            # Ensure right context is non-empty before adding
            if index[i] + 1 < x.shape[1]:
                right_contexts.append(x[i, index[i]+1:])
            else:
                right_contexts.append(torch.zeros(1, x.shape[2], device=x.device))  # Dummy zero tensor

        # Pad sequences to match max length in batch
        left_context = torch.nn.utils.rnn.pad_sequence(left_contexts, batch_first=True)
        right_context = torch.nn.utils.rnn.pad_sequence(right_contexts, batch_first=True)
        

        # Process through LSTMs
        left_out, _ = self.left_lstm(left_context)
        right_out, _ = self.right_lstm(right_context)

        aspect_out, _ = self.aspect_lstm(aspect.unsqueeze(1))
        combined_rep = torch.cat([left_out.mean(dim=1), right_out.mean(dim=1), aspect_out.squeeze(1)], dim=-1)
        return self.fc(self.dropout(combined_rep))

# Hybrid BiLSTM-GRU Model
class HybridBiLSTMGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout_rate=0.6):
        super(HybridBiLSTMGRU, self).__init__()
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)

        # Attention layers
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.aspect_attention = nn.Linear(hidden_dim * 2, 1)  

        # Feature processing
        self.layer_norm = nn.LayerNorm(hidden_dim * 4)  # Better for NLP
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self.dropout = nn.Dropout(dropout_rate)  # Regularization

    def forward(self, x, aspect, index, lengths):
        # Ensure lengths are on CPU
        lengths = lengths.cpu()

        # Packing for efficiency
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed_x)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        gru_out, _ = self.gru(lstm_out)

        # Context-aware attention
        attn_weights = torch.tanh(self.attention(gru_out))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)

        # Aspect-aware attention
        aspect_vector = gru_out[torch.arange(gru_out.size(0)), index]
        aspect_attn_weights = torch.tanh(self.aspect_attention(aspect_vector))
        aspect_attn_weights = torch.softmax(aspect_attn_weights, dim=1)
        aspect_vector = aspect_attn_weights * aspect_vector

        # Concatenation instead of addition
        combined_vector = torch.cat([context_vector, aspect_vector], dim=1)
        combined_vector = F.relu(self.layer_norm(combined_vector))  # Apply LayerNorm + ReLU
        combined_vector = self.dropout(combined_vector)  # Apply Dropout

        output = self.fc(combined_vector)
        return output

#Save Model Checkpoint    
def save_model(model, model_name):
    torch.save(model.state_dict(), f"{model_name}.pth")        
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig("loss_plot.png")
    print("📉 Loss plot saved as 'loss_plot.png'")
# Training function
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2, verbose=True)

    train_losses, val_losses = [], []
    model.to(DEVICE)  # Ensure model is on the correct device

    for epoch in range(10):  # Define EPOCHS
        model.train()
        total_loss, correct, total = 0, 0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/10", unit="batch") as pbar:
            for inputs, aspects, indices, labels, lengths in pbar:
                inputs, aspects, indices, labels, lengths = (
                    inputs.to(DEVICE),
                    aspects.to(DEVICE),
                    indices.to(DEVICE),
                    labels.to(DEVICE),
                    lengths.to(DEVICE),
                )
                
                optimizer.zero_grad()
                outputs = model(inputs, aspects, indices, lengths)  

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix(loss=loss.item(), accuracy=correct / total if total > 0 else 0)

        train_losses.append(total_loss / len(train_loader))
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", unit="batch") as pbar:
                for inputs, aspects, indices, labels, lengths in pbar:
                    inputs, aspects, indices, labels, lengths = (
                        inputs.to(DEVICE),
                        aspects.to(DEVICE),
                        indices.to(DEVICE),
                        labels.to(DEVICE),
                        lengths.to(DEVICE),
                    )

                    outputs = model(inputs, aspects, indices, lengths)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

                    pbar.set_postfix(loss=loss.item(), accuracy=val_correct / val_total if val_total > 0 else 0)

        val_losses.append(val_loss / len(val_loader))
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        scheduler.step(val_loss)  # Reduce LR based on validation loss
    plot_losses(train_losses,val_losses)
    return model

# Inference function
def inference(model_path, test_file):
    model = HybridBiLSTMGRU(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, 0.5).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_data = load_and_preprocess(test_file)
    test_loader = DataLoader(ABSADataset(test_data), batch_size=16, shuffle=False, collate_fn=collate_fn)
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, aspects, indices, labels, lengths in test_loader:
            inputs, aspects, indices, labels, lengths = (
                inputs.to(DEVICE), aspects.to(DEVICE), indices.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            )
            outputs = model(inputs, aspects, indices, lengths)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")
    
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 64
    NUM_CLASSES = 4  # Positive, Negative, Neutral, Conflict
    # NUM_HEADS = 8
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.6
    train_loader, val_loader = prepare_dataset("train.json", "val.json")
    # # model = TD_LSTM_Attention(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    model = HybridBiLSTMGRU(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES,DROPOUT_RATE)
    trained_model = train_model(model, train_loader, val_loader)
    save_model(train_model, "best_model_checkpoint")