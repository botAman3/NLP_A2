from datasets import load_dataset 
from transformers import AutoTokenizer , AutoModelForQuestionAnswering , TrainingArguments , Trainer , AutoModel
import torch
import torch.nn as nn 
from matplotlib import pyplot as plt 
from torch.optim import AdamW
from torch.utils.data import DataLoader  , Dataset
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define SpanBERT-CRF model

def preprocess(data):
    inputs = tokenizer(
        data["question"] , data["context"],
        truncation=True , padding="max_length" , max_length=384
    )

    start_pos , end_pos = [] , []


    for i , (ans , context) in enumerate(zip(data["answers"] , data["context"])):
        if len(ans["text"]) == 0:
            start_pos.append(0)
            end_pos.append(0)
        else :
            start_idx = context.find(ans["text"][0])
            end_idx = start_idx + len(ans["text"][0])

            start_pos.append(start_idx)
            end_pos.append(end_idx)

    inputs["start_positions"] = start_pos
    inputs["end_positions"] = start_pos

    return inputs


def plot_losses(train_losses):
    
    train_loss = [log["loss"] for log in train_losses  if "loss" in log]
    val_loss = [log["eval_loss"] for log in train_losses if "eval_loss" in log]


    plt.plot(train_loss , label = "Traing Loss" , marker="O")
    plt.plot(train_loss , label = "Validation Loss" , marker="X")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()



def trainSpanBERT(model , training_arguments , train_dataset , validation_dataset):
    trainer = Trainer(
        model=model,
        args= training_arguments,
        train_dataset= train_dataset,
        eval_dataset=validation_dataset
        
    )

    hist = trainer.train()

    train_losses = trainer.state.log_history

    
    plot_losses(train_losses)


    model.save_pretrained("./trained_spanBERT")
    tokenizer.save_pretrained("./trained_spanBERT")



#========================================================SpanBERT-CRF-Training=====================================================================================


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

# Custom dataset class
class SquadDataset(Dataset):
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

        encoding = self.tokenizer(context, question, truncation=True, padding="max_length", max_length=self.max_length)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        labels = [0] * self.max_length  # Initialize all as 'O' (outside)
        
        if item["answers"]["text"]:
            answer_text = item["answers"]["text"][0]
            start_char = item["answers"]["answer_start"][0]
            start_token = encoding.char_to_token(start_char)
            end_token = encoding.char_to_token(start_char + len(answer_text) - 1)

            if start_token is not None and end_token is not None:
                labels[start_token] = 1  # 'B' tag
                for i in range(start_token + 1, end_token + 1):
                    labels[i] = 2  # 'I' tag
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# Training function
def train_model(model, train_loader, val_loader, optimizer, epochs=6):
    train_losses, val_losses, em_scores = [], [], []
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_losses.append(total_loss / len(train_loader))

        # Validation Loop
        model.eval()
        val_loss, predictions, references = 0, [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                loss = model(input_ids, attention_mask, labels)
                val_loss += loss.item()
                preds = model(input_ids, attention_mask)
                
                predictions.extend(preds)
                references.extend(labels.cpu().numpy().tolist())
        
        em_score = exact_match_score(predictions, references)
        val_losses.append(val_loss / len(val_loader))
        em_scores.append(em_score)

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, EM Score: {em_score:.4f}")
        torch.save(model.state_dict(), f"spanbert_crf_epoch{epoch+1}.pt")
    
    plot_results(train_losses, val_losses, em_scores)

# Exact Match Score Calculation
def exact_match_score(predictions, references):
    matches = sum((set(p) == set(r)) for p, r in zip(predictions, references))
    return (matches / len(references)) * 100

# Plot Loss & EM Score
def plot_results(train_losses, val_losses, em_scores):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    
    plt.figure()
    plt.plot(em_scores, label='EM Score', color='green')
    plt.xlabel("Epochs")
    plt.ylabel("EM Score")
    plt.legend()
    plt.savefig("em_score_plot.png")



if __name__ == "__main__":

    dataset = load_dataset("squad_v2")

    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")

    train_data = dataset['train'].select(range(15000))
    val_data = dataset['validation']

    model = AutoModelForQuestionAnswering.from_pretrained("SpanBERT/spanbert-large-cased")

    training_arguments = TrainingArguments(
        output_dir="./qa",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate= 3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        logging_dir="./logs",
        logging_steps=100
    )

    trainSpanBERT(model , training_arguments , train_data , val_data)

###=============================================================SpanBERT-CRF=============================================================================### 
    train_dataset = SquadDataset(train_data, tokenizer)
    val_dataset = SquadDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    model = SpanBERT_CRF("SpanBERT/spanbert-base-cased", num_tags=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train_model(model, train_loader, val_loader, optimizer)





