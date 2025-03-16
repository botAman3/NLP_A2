import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_scheduler , AutoModel
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from torchcrf import CRF
import json

def exact_match_score(predictions, references):
    assert len(predictions) == len(references), "Lists must have the same length"
    matches = sum(p == r for p, r in zip(predictions, references))
    return matches / len(references) * 100

# Load SQuAD v2 dataset

# +++++++++++++++++++++++++++++++++++++++++++++++SpanBERT MODEL START +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        question = item["question"]
        answers = item["answers"]
        
        encoding = self.tokenizer(
            context, question,
            truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Extract answer positions
        if answers["answer_start"]:  # Ensure answer exists
            answer_start = answers["answer_start"][0]
            answer_text = answers["text"][0]

            # Convert character positions to token positions
            start_positions = encoding.char_to_token(answer_start)
            end_positions = encoding.char_to_token(answer_start + len(answer_text) - 1)

            # Ensure valid positions
            if start_positions is None or end_positions is None:
                start_positions, end_positions = 0, 0
        else:
            start_positions, end_positions = 0, 0  # No answer

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
        }
    

# Training function
def trainSpanBERT(model, train_loader, val_loader, optimizer, scheduler, num_epochs=6):
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, em_scores = [], [], []
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            loss = criterion(start_logits, start_positions) + criterion(end_logits, end_positions)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_pred_start, all_pred_end = [], []
        all_true_start, all_true_end = [], []
        contexts = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                loss = criterion(start_logits, start_positions) + criterion(end_logits, end_positions)
                total_val_loss += loss.item()
                
                pred_start = torch.argmax(start_logits, dim=1).cpu().tolist()
                pred_end = torch.argmax(end_logits, dim=1).cpu().tolist()
                
                all_pred_start.extend(pred_start)
                all_pred_end.extend(pred_end)
                all_true_start.extend(start_positions.cpu().tolist())
                all_true_end.extend(end_positions.cpu().tolist())
                contexts.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids.cpu()])
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        em_score = exact_match_score(all_pred_start, all_true_start)
        em_scores.append(em_score)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, EM Score = {em_score:.2f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_spanbert_model.pth")
    
    # Plot losses and EM scores
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), em_scores, label='EM Score', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Exact Match Score')
    plt.legend()
    plt.title('Validation EM Score')
    
    plt.savefig('training_results_SpanBERT.png')
    plt.show()# Prepare data, loaders, optimizer, and scheduler


# +++++++++++++++++++++++++++++++++++++++++++++++SpanBERT MODEL END +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++









# +++++++++++++++++++++++++++++++++++++++++++++++SpamBERT_CRF MODEL START +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class SpanBERT_CRF(nn.Module):
    def __init__(self, model_name, num_tags):
        super(SpanBERT_CRF, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size
        self.fc = nn.Linear(self.hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
            return predictions

# Load and preprocess SQuAD v2 dataset
def load_squad(path):
    with open(path, "r") as f:
        data = json.load(f)["data"]
    examples = []
    for article in data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answers = [ans["text"] for ans in qa["answers"]]
                examples.append((context, question, answers))
    return examples

# Custom dataset class
class SquadDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        question = item["question"]

        encoding = self.tokenizer(context, question, truncation=True, padding="max_length", max_length=512)

        if item["answers"]["text"]:
            answer_text = item["answers"]["text"][0]
            start_char = item["answers"]["answer_start"][0]  # Get start char index

            
            start_token = encoding.char_to_token(start_char)
            end_token = encoding.char_to_token(start_char + len(answer_text) - 1)

            
            if start_token is None:
                start_token = 0  

            if end_token is None:
                end_token = start_token 
        else:
            answer_text = ""
            start_token, end_token = 0, 0  
        
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor([start_token, end_token], dtype=torch.long)
        }

    # Training function
def train_SpanBERT_CRF(model, train_loader, val_loader, optimizer, epochs=6):
    train_losses, val_losses, em_scores = [], [], []
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_em = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size, seq_length = input_ids.shape
            labels_start, labels_end = labels[:, 0], labels[:, 1]
            
            seq_labels = torch.full((batch_size, seq_length), 0, dtype=torch.long, device=device)
            for i in range(batch_size):
                start, end = labels_start[i].item(), labels_end[i].item()
                if 0 <= start < seq_length:
                    seq_labels[i, start] = 1
                if 0 <= end < seq_length:
                    seq_labels[i, end] = 2
            
            loss = model(input_ids, attention_mask, seq_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss, val_em = 0, 0
        all_predictions, all_answers = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                batch_size, seq_length = input_ids.shape
                seq_labels = torch.full((batch_size, seq_length), 0, dtype=torch.long, device=device)
                labels_start, labels_end = labels[:, 0], labels[:, 1]
                
                for i in range(batch_size):
                    start, end = labels_start[i].item(), labels_end[i].item()
                    if 0 <= start < seq_length:
                        seq_labels[i, start] = 1
                    if 0 <= end < seq_length:
                        seq_labels[i, end] = 2
                
                loss = model(input_ids, attention_mask, seq_labels)
                val_loss += loss.item()
                
                predictions = model(input_ids, attention_mask)
                all_predictions.extend(predictions)
                all_answers.extend(labels.cpu().numpy().tolist())
        
        print(all_predictions , all_answers)
                
        em_score = exact_match_score(all_predictions, all_answers)
        val_losses.append(val_loss / len(val_loader))
        em_scores.append(em_score)

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, EM Score: {em_score:.4f}")

        torch.save(model.state_dict(), f"spanbert_crf_epoch{epoch+1}.pt")
    
    # Plot Loss & EM Score
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("loss_plot.png")
    plt.show()
    
    plt.figure()
    plt.plot(range(1, epochs+1), em_scores, label='EM Score', color='green')
    plt.xlabel("Epochs")
    plt.ylabel("EM Score")
    plt.legend()
    plt.title("Exact Match Score Over Epochs")
    plt.savefig("em_score_plot.png")
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++SpamBERT_CRF MODEL END +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


dataset = load_dataset("squad_v2")


MODEL_NAME = "SpanBERT/spanbert-large-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


train_data = dataset['train'].select(range(15000))
val_data = dataset['validation']

train_dataset = QADataset(train_data, tokenizer)
val_dataset = QADataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=3e-5)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 6)

# Train model
trainSpanBERT(model, train_loader, val_loader, optimizer, scheduler)



train_dataset = SquadDataset(train_data, tokenizer)
val_dataset = SquadDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = SpanBERT_CRF("SpanBERT/spanbert-base-cased", num_tags=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_SpanBERT_CRF(model, train_loader, val_loader, optimizer)

