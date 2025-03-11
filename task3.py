from datasets import load_dataset 
from transformers import AutoTokenizer , AutoModelForQuestionAnswering , TrainingArguments , Trainer , AutoModel
import torch
import torch.nn as nn 

from torch.optim import AdamW
from torch.utils.data import DataLoader 
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpanBERT_CRF(nn.Module):
    def __init__ (self):
        super(SpanBERT_CRF , self).__init__()
        self.spanbert = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")
        hidden_size = self.spanbert.config.hidden_size

        self.start_classifier = nn.Linear(hidden_size , 1)
        self.end_classifier = nn.Linear(hidden_size , 1)



        self.crf = torchcrf.CRF(num_tags = 2 , batch_first = True)

    def forward(self , input_ids , attention_mask , start_pos = None , end_pos = None):
        output = self.spanbert(input_ids , attention_mask=attention_mask)

        seq_output = output.last_hidden_state 


        start_log = self.start_classifier(seq_output).squeeze(-1)
        end_log = self.end_classifier(seq_output).squeeze(-1)


        logits = torch.stack([start_log , end_log] , dim=-1)


        if start_pos is not None and end_pos is not None:

            targets = torch.stack([start_pos , end_pos] , dim=-1)
            loss =- self.crf(logits , targets , mask=attention_mask.byte())

            return loss 
        else :
            return self.crf.decode(logits)


def train_model(model, train_dataloader, optimizer, epochs=10):
    
    model.to(device)

    for epoch in range(epochs):
        model.train()

        total_loss = 0 

        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)


            loss = model(input_ids , attention_mask , start_pos , end_pos)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()
    pred , ref = [] , []
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            pred_start , pred_end = model(input_ids , attention_mask)

        pred_ans = [
            tokenizer.decode(input_ids[i][pred_start[i] : pred_end[i]]) for i in range(len(input_ids)) 
        ]


        true_ans = [a["text"][0] if len(a["text"]) > 0 else "" for a in batch["answers"]]

        pred.extend(pred_ans)
        ref.extend(true_ans)


    return pred , ref 


def exact_match_score(predictions, references):
    assert len(predictions) == len(references), "Lists must have the same length"
    matches = sum(p == r for p, r in zip(predictions, references))
    return matches / len(references) * 100  


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
    inputs["end_position"] = start_pos

    return inputs


if __name__ == "__main__":

    dataset = load_dataset("squad_v2")

    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")

    train_dataset = dataset["train"].select(range(15000))
    train_dataset = train_dataset.map(preprocess , batched=True)

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



    trainer = Trainer(
        model=model,
        args= training_arguments,
        train_dataset= train_dataset
    )

    trainer.train()

    model = SpanBERT_CRF()
    optimizer = AdamW(model.parameters() , lr=3e-5)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    pred , ref = train_model(model , train_dataloader , optimizer)

    em_score = exact_match_score(pred, ref)
    print(f"Exact-Match Score: {em_score:.2f}%")





