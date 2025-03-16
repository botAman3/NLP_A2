import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import gensim.downloader as api
from conlleval import evaluate
from tqdm import tqdm
import gensim.downloader as api
from matplotlib import pyplot as plt

def load_pretrained_embeddings( word2idx, embedding_dim , model_name = 'glove-wiki-gigaword-300'):
    """Loads embeddings from gensim and maps them to the vocabulary."""
    embedding_model = api.load(model_name)
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))

    for word, idx in word2idx.items():
        if word in embedding_model:
            embeddings[idx] = embedding_model[word]

    return torch.tensor(embeddings, dtype=torch.float32)

def load_fasttext_embeddings(word2idx, embedding_dim=300):
    fasttext_model = api.load("fasttext-wiki-news-subwords-300")
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))
    for word, idx in word2idx.items():
        if word in fasttext_model:
            embeddings[idx] = fasttext_model[word]
    return torch.tensor(embeddings, dtype=torch.float32)

class ATE_Dataset(Dataset):
    def __init__(self, sentences, labels, word2idx, label2idx, max_len=100):
        self.sentences = [[word2idx.get(word, word2idx['<UNK>']) for word in sent] for sent in sentences]
        self.labels = [[label2idx.get(label, 0) for label in lab] for lab in labels]
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sent = self.sentences[idx]
        label = self.labels[idx]
        pad_len = max(0, self.max_len - len(sent))
        sent = sent[:self.max_len] + [0] * pad_len
        label = label[:self.max_len] + [0] * pad_len
        return torch.tensor(sent, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class RNN_ATE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(RNN_ATE, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

class GRU_ATE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(GRU_ATE, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer,name,epochs=10):
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).permute(0, 2, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(train_loader))
        
        train_losses.append(total_loss / len(train_loader))
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).permute(0, 2, 1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    # **Plot Training & Validation Loss**
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.savefig(f'{name}.png')

    return train_losses, val_losses

def build_label2idx(labels):
    unique_labels = set(label for seq in labels for label in seq)
    return {label: idx for idx, label in enumerate(unique_labels, start=0)}

def build_word2idx(sentences, min_freq=1):
    word_counts = Counter(word for sentence in sentences for word in sentence)
    word2idx = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    return word2idx

def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = [entry["tokens"] for entry in data.values()]
    labels = [entry["labels"] for entry in data.values()]
    return sentences, labels


def evaluate_model(model, data_loader, idx2label, device="cpu"):
    model.eval()
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=2).cpu().numpy()
            targets = targets.cpu().numpy()

            for i in range(len(inputs)):
                true_seq, pred_seq = [], []
                for j in range(len(inputs[i])):
                    if inputs[i][j].item() == 0:  # Ignore padding
                        continue
                    true_seq.append(idx2label[targets[i][j]])
                    pred_seq.append(idx2label[predictions[i][j]])
                
                true_labels.append(true_seq)
                pred_labels.append(pred_seq)

    # **Ensure labels follow IOB2 format**
    true_labels_flat = [label if "-" in label or label == "O" else f"I-{label}" for seq in true_labels for label in seq]
    pred_labels_flat = [label if "-" in label or label == "O" else f"I-{label}" for seq in pred_labels for label in seq]

    # Evaluate using conlleval
    chunk_result = evaluate(true_labels_flat, pred_labels_flat, verbose=True)

    print("\nChunk-Level Performance:")
    print(f"Precision: {chunk_result[0]:.2f}, Recall: {chunk_result[1]:.2f}, F1-Score: {chunk_result[2]:.2f}")

    return chunk_result

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sentences_train, labels_train = load_dataset("train_task 1.json")
    sentences_val, labels_val = load_dataset("val_task 1.json")
    
    word2idx = build_word2idx(sentences_train)
    label2idx = build_label2idx(labels_train)
    
    glove_embeddings = load_pretrained_embeddings(word2idx, 300)
    fasttext_embeddings = load_fasttext_embeddings(word2idx, 300)
    
    train_dataset = ATE_Dataset(sentences_train, labels_train, word2idx, label2idx)
    val_dataset = ATE_Dataset(sentences_val, labels_val, word2idx, label2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    models = {
        "RNN_GloVe": RNN_ATE(len(word2idx), 300, 128, 3, glove_embeddings),
        "GRU_GloVe": GRU_ATE(len(word2idx), 300, 128, 3, glove_embeddings),
        "RNN_FastText": RNN_ATE(len(word2idx), 300, 128, 3, fasttext_embeddings),
        "GRU_FastText": GRU_ATE(len(word2idx), 300, 128, 3, fasttext_embeddings)
    }
    
    criterion = nn.CrossEntropyLoss()
    
    for name, model in models.items():
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"Training {name}...")
        train_model(model, train_loader, val_loader, criterion, optimizer, name ,epochs=10 )
        torch.save(model.state_dict(), f"{name}.pth")
        print(f"Saved {name} model.")
    
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    for name, model in models.items():
        model.load_state_dict(torch.load(f"{name}.pth"))
        model.to(device)
        print(f"Evaluating {name}...")
        evaluate_model(model, val_loader, idx2label, device=device)
