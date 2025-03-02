import json 
import numpy as np 
import spacy 
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader , Dataset
from collections import Counter
import gensim.downloader as api
from conlleval import evaluate
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")


def bio_tagging(tokens, entities):
    tags = ["O"] * len(tokens) 
    
    for entity in entities:
        entity_tokens = entity.split()
        start_idx = -1
        
        # Find entity position
        for i in range(len(tokens)):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                start_idx = i
                break
        
        if start_idx != -1:
            tags[start_idx] = "B"  # Mark beginning
            for j in range(1, len(entity_tokens)):
                tags[start_idx + j] = "I"  # Mark inside
                
    return tags


class ATE_Dataset(Dataset):
    def __init__(self, sentences, labels, word2idx, label2idx, max_len=100):
        self.sentences = [[word2idx.get(word, word2idx['<UNK>']) for word in sent] for sent in sentences]
        self.labels = [[label2idx.get(label, 0) for label in lab] for lab in labels]  # Convert labels to indices
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sent = self.sentences[idx]
        label = self.labels[idx]
        
        # Padding
        pad_len = max(0, self.max_len - len(sent))
        sent = sent[:self.max_len] + [0] * pad_len
        label = label[:self.max_len] + [0] * pad_len  # Ensure labels are also padded
        
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

def preprocessData(jsonData):
    preprocessedData = {}
    for data in jsonData:
        entities = []
        tokens = (data['sentence']).split()
        for f in data['aspect_terms']:
            entities.append(f['term'])
        bioTags = bio_tagging(tokens , entities)


        variables = {
            "sentence" : data['sentence'],
            "tokens"   : tokens ,
            "labels"   : bioTags ,
            "aspect_terms" : entities
        }

        preprocessedData[data['sentence_id']] = variables

    return preprocessedData

def load_pretrained_embeddings(embedding_path, word2idx, embedding_dim):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))
    
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype=np.float32)
                if len(vector) == embedding_dim and word in word2idx:
                    embeddings[word2idx[word]] = vector
            except ValueError:
                print(f"Skipping malformed line: {line[:50]}...")  # Print first 50 chars of the bad line
    
    return torch.tensor(embeddings, dtype=torch.float32)
    

    
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
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
    
    return train_losses, val_losses


def predict_aspect_terms(model, sentence, word2idx, idx2label, max_len=100, device='cpu'):
    model.eval()
    
    # Tokenize and convert words to indices
    sentence_tokens = sentence.split()  # Ensure tokenization matches training
    input_ids = [word2idx.get(word, word2idx['<UNK>']) for word in sentence_tokens]

    # Pad sequence
    pad_len = max(0, max_len - len(input_ids))
    input_ids = input_ids[:max_len] + [0] * pad_len  # Pad or truncate

    # Convert to tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.argmax(output, dim=2).cpu().numpy()[0]  # Get highest-probability labels

    predicted_labels = [idx2label[idx] for idx in predictions[:len(sentence_tokens)]]

    
    aspect_terms = []
    current_term = []

    for word, label in zip(sentence_tokens, predicted_labels):
        if label == "B":  # Beginning of an aspect
            if current_term:
                aspect_terms.append(" ".join(current_term))  # Store previous aspect
            current_term = [word]  # Start new aspect
        elif label == "I":  # Inside an aspect
            current_term.append(word)
        else:
            if current_term:
                aspect_terms.append(" ".join(current_term))  # Store aspect term
                current_term = []

    if current_term:
        aspect_terms.append(" ".join(current_term))  # Store last aspect term

    return aspect_terms


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # with open("train.json" , "r") as file :
    #     train_json = json.load(file)

    # with open("val.json" , "r") as file :
    #     val_json = json.load(file)

    # preprocessed_train = preprocessData(train_json)
    # preprocessed_val = preprocessData(val_json)

    # with open("train_task 1.json", "w") as file:
    #     json.dump(preprocessed_train, file, indent=4)

    # with open("val_task 1.json", "w") as file:
    #     json.dump(preprocessed_val, file, indent=4)

    # print("JSON file saved successfully!")
    sentences_train, labels_train = load_dataset("train_task 1.json")
    sentences_val , labels_val = load_dataset("val_task 1.json")
    word2idxTrain = build_word2idx(sentences_train)
    label2idxTrain = build_label2idx(labels_train)


    word2idx_Val = build_word2idx(sentences_val)
    label2idxVal = build_label2idx(labels_val)
    

    embedding_dim = 300
    embedding_matrix = load_pretrained_embeddings("glove.840B.300d.txt", word2idxTrain, embedding_dim)



    train_dataset = ATE_Dataset(sentences_train, labels_train, word2idxTrain, label2idxTrain)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = ATE_Dataset(sentences_val, labels_val, word2idx_Val, label2idxVal)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    
    model = GRU_ATE(vocab_size=len(word2idxTrain), embedding_dim=embedding_dim, hidden_dim=128, output_dim=3, pretrained_embeddings=embedding_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)


    idx2label = {idx: label for label, idx in label2idxTrain.items()}


    sentence = "The food was delicious but the service was slow."
    predicted_aspects = predict_aspect_terms(model, sentence, word2idxTrain, idx2label, device=device)

    print("Predicted Aspect Terms:", predicted_aspects)


    

