import json
import spacy
from spacy.tokenizer import Tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Modify tokenizer to keep contractions together
infixes = nlp.Defaults.infixes
infixes = [x for x in infixes if x not in [r"\'"]]
nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=None)

# Function to load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if len(values) != EMBEDDING_DIM + 1:  # Ensure valid line
                continue
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype=np.float32)
                embeddings[word] = vector
            except ValueError:
                print(f"Skipping corrupted line: {line}")  # Debugging output
    return embeddings

# Function to get word embedding
def get_embedding(word):
    return torch.tensor(word_vectors[word], dtype=torch.float32) if word in word_vectors else torch.zeros(EMBEDDING_DIM)

# Normalize token to remove apostrophes
def normalize_token(token):
    return token.replace("'", "")

# Data preprocessing
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
                    "aspect term": aspect_term,
                    "index": index
                })

    return preprocessed_data

# Load and preprocess dataset
def load_and_preprocess(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return preprocess_data(json_data)

# Custom PyTorch Dataset
class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"]
        aspect_term = sample["aspect term"] if isinstance(sample["aspect term"], str) else sample["aspect term"][0]
        index = sample["index"]
        polarity = sample["polarity"]

        token_embeddings = torch.stack([get_embedding(tok) for tok in tokens])

        polarity_map = {"positive": 2, "negative": 0, "neutral": 1, "conflict": 3}
        polarity_label = polarity_map[polarity]

        return token_embeddings, index, polarity_label

# Function to collate batch with padding
def collate_fn(batch):
    sequences, indices, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(indices), torch.tensor(labels)

# Prepare DataLoaders
def prepare_dataset(train_path, val_path):
    train_data = load_and_preprocess(train_path)
    val_data = load_and_preprocess(val_path)
    train_dataset = ABSADataset(train_data)
    val_dataset = ABSADataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print("Dataset Prepared")
    return train_loader, val_loader

# BiLSTM with Aspect-Specific Attention Model
class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)  # Dropout for regularization
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, index):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output

# Evaluate model
def evaluate_model(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, indices, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs, indices)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

# Training function
def train_model(train_loader, val_loader):
    model = BiLSTMAttention(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, indices, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, indices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model

# Save model checkpoint
def save_checkpoint(model, filename="bilstm_attention.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model checkpoint saved: {filename}")

# Load model checkpoint
def load_checkpoint(filename="bilstm_attention.pth"):
    model = BiLSTMAttention(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(filename, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from checkpoint: {filename}")
    return model

# Inference function
def predict(model, sentence):
    tokens = sentence.split()
    embeddings = torch.stack([get_embedding(tok) for tok in tokens]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(embeddings, torch.tensor([0]).to(DEVICE))
        _, predicted = torch.max(output, 1)

    polarity_map = {0: "negative", 1: "neutral", 2: "positive"}
    return polarity_map[predicted.item()]

# Main Execution
if __name__ == "__main__":
    EMBEDDING_PATH = "/home/aasif057/nlp_embeddings/glove.840B.300d.txt"
    EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES = 300, 256, 4
    BATCH_SIZE, EPOCHS, LEARNING_RATE = 32, 10, 0.0005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word_vectors = load_glove_embeddings(EMBEDDING_PATH)

    train_loader, val_loader = prepare_dataset("train.json", "val.json")
    trained_model = train_model(train_loader, val_loader)
