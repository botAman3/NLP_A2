import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Function to get BERT word embedding
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state.squeeze(0)

# Load and preprocess dataset
def load_and_preprocess(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    preprocessed_data = []
    for data in json_data:
        sentence = data['sentence']
        aspect_terms = data.get('aspect_terms', [])
        for aspect in aspect_terms:
            aspect_term = aspect['term']
            polarity = aspect['polarity']
            preprocessed_data.append({
                "sentence": sentence,
                "aspect_term": aspect_term,
                "polarity": polarity
            })
    return preprocessed_data

# Custom PyTorch Dataset
class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sentence = sample["sentence"]
        aspect_term = sample["aspect_term"]
        polarity = sample["polarity"]
        sentence_embedding = get_embedding(sentence)
        aspect_embedding = get_embedding(aspect_term)
        polarity_map = {"positive": 2, "negative": 0, "neutral": 1, "conflict": 3}
        polarity_label = polarity_map[polarity]
        return sentence_embedding, aspect_embedding, polarity_label

# Function to collate batch with padding
def collate_fn(batch):
    sequences, aspects, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    padded_aspects = pad_sequence(aspects, batch_first=True)
    return padded_sequences, padded_aspects, torch.tensor(labels)

# Prepare DataLoaders
def prepare_dataset(train_path, val_path):
    train_data = load_and_preprocess(train_path)
    val_data = load_and_preprocess(val_path)
    train_dataset = ABSADataset(train_data)
    val_dataset = ABSADataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

# BiLSTM with Aspect-Specific Attention Model
class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.6)  # Increased dropout
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, index):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = 0
        self.counter = 0

    def should_stop(self, val_acc):
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.counter = 0  # Reset counter if validation accuracy improves
        else:
            self.counter += 1  # Increment counter if no improvement
        return self.counter >= self.patience  # Stop training if counter exceeds patience

def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, indices, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs, indices)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0

def train_model(train_loader, val_loader):
    model = BiLSTMAttention(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)  # Increased weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Reduce LR every 3 epochs
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

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
        scheduler.step()  # Adjust learning rate

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping check
        if early_stopping.should_stop(val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model

# Main Execution
if __name__ == "__main__":
    EMBEDDING_DIM = 768  # BERT embedding size
    HIDDEN_DIM = 256
    NUM_CLASSES = 4  # Positive, Negative, Neutral, Conflict
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = prepare_dataset("train.json", "val.json")
    trained_model = train_model(train_loader, val_loader)
