import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
import spacy
from spacy.tokenizer import Tokenizer
import matplotlib.pyplot as plt

def get_embedding(word):
    return torch.tensor(word_vectors[word], dtype=torch.float32) if word in word_vectors else torch.zeros(EMBEDDING_DIM)

def normalize_token(token):
    return token.replace("'", "")

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
            aspect_tokens = [token.text for token in aspect_doc]

            index = -1
            for i in range(len(tokens) - len(aspect_tokens) + 1):
                if tokens[i:i + len(aspect_tokens)] == aspect_tokens:
                    index = i
                    break

            if index != -1:
                preprocessed_data.append({
                    "tokens": tokens,
                    "polarity": polarity,
                    "aspect_term": [aspect_term],
                    "index": index
                })

    return preprocessed_data

def save_preprocessed_data(file_path,json_data):
    save_path = file_path.replace(".json","") + "_task_2.json"
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4)

def load_and_preprocess(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    preprocessed_data = preprocess_data(json_data)
    # save_preprocessed_data(file_path,preprocessed_data)
    return preprocessed_data

# Custom PyTorch Dataset
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

        token_embeddings = torch.stack([get_embedding(tok) for tok in tokens])
        aspect_embedding = get_embedding(aspect_term)

        polarity_map = {"positive": 2, "negative": 0, "neutral": 1, "conflict": 3}
        polarity_label = polarity_map[polarity]

        return token_embeddings, aspect_embedding, index, polarity_label

# Function to collate batch with padding
def collate_fn(batch):
    sequences, aspects, indices, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.stack(aspects), torch.tensor(indices), torch.tensor(labels)

# Prepare DataLoaders
def prepare_dataset(train_path, val_path):
    train_data = load_and_preprocess(train_path)
    val_data = load_and_preprocess(val_path)
    train_dataset = ABSADataset(train_data)
    val_dataset = ABSADataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

# Hybrid BiLSTM-GRU with Aspect-Specific Attention model
class HybridBiLSTMGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(HybridBiLSTMGRU, self).__init__()
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.aspect_attention = nn.Linear(hidden_dim * 2, 1)  # Aspect-Specific Attention
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, aspect, index):
        lstm_out, _ = self.bilstm(x)
        gru_out, _ = self.gru(lstm_out)
        
        # Compute attention scores for the whole sentence
        attention_weights = torch.tanh(self.attention(gru_out))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        
        # Compute aspect-based attention
        aspect_vector = gru_out[torch.arange(gru_out.size(0)), index]

        # Combine both context and aspect-aware representations
        combined_vector = context_vector + aspect_vector
        output = self.fc(combined_vector)
        return output

def save_model(model, model_name):
    torch.save(model.state_dict(), f"{model_name}.pth")

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    
# Training function with tqdm for progress tracking
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        for inputs, aspects, indices, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, aspects, indices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_losses.append(total_loss / len(train_loader))
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, aspects, indices, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs, aspects, indices)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_losses.append(val_loss / len(val_loader))
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    # save_model(model)
    plot_losses(train_losses, val_losses)
    return model


# Main Execution
if __name__ == "__main__":
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Modify the tokenizer to keep contractions together
    infixes = nlp.Defaults.infixes
    infixes = [x for x in infixes if x not in [r"\'"]]
    nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=None)

    # Load pre-trained FastText embeddings
    EMBEDDING_PATH = "/home/aasif057/nlp_embeddings/cc.en.300.vec"
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    NUM_CLASSES = 4  # Positive, Negative, Neutral, Conflict
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FastText word vectors
    word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False)
    
    #Initialize the model
    model = HybridBiLSTMGRU(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    
    train_loader, val_loader = prepare_dataset("train.json", "val.json")
    trained_model = train_model(model, train_loader, val_loader)
    model_name = "bilstm_gru"
    save_model(trained_model,model_name)