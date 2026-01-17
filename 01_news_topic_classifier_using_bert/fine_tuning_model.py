"""
Problem Statement: For the huge amount of data in this internet era, it is difficult to catogarize thoes data/news.

Objective: Build a news classifier by fine tuining a BERT model on Agnews dataset
"""


# Neccessary imports
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score


# Load the Agnews dataset
dataset = load_dataset("ag_news")

# Tokenixer instance and a function for tokenizing texts
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenizer_function(text):
    return tokenizer(
        text["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenizing the Agnews dataset
tokenized_datasets = dataset.map(tokenizer_function, batched=True)


# Preparing dataset for Pytorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# initalizing model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# function that evaluate model
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1
    }


# training arguments and settings
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs"
)

# initializong trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# start training the model
trainer.train()

# evalutaing the model
results = trainer.evaluate()
print(results)


# testing model by giving a data
text = "Elon Musk launching a free Starlink services in Iran."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

pred = torch.argmax(outputs.logits, dim=1).item()

label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

label_map[pred]