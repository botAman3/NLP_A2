from datasets import load_dataset 
from transformers import AutoTokenizer , AutoModelForQuestionAnswering , TrainingArguments , Trainer



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

    model = AutoModelForQuestionAnswering("SpanBERT/spanbert-large-cased")

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

    trainer.train