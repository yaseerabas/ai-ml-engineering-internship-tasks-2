import torch
import numpy as np
from transformers import pipeline, AutoTokenizer,AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

import json

# Loading dataset form the json
with open("example_ticket_support_dataset.json", "r") as f:
    data = json.load(f)

tickets = data["tickets"]
categories = data["categories"]
labels = data["labels"]

# prepare label mapping
label2id = {label: i for i, label in enumerate(categories)}
id2label = {i: label for label, i in label2id.items()}

y = [label2id[l] for l in labels]

# init tokenizer and performing tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

encodings = tokenize(tickets)


# dataset class
class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = TicketDataset(encodings, y)


# load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(categories),
    id2label=id2label,
    label2id=label2id
)

# evaluate metrics
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(pred.label_ids, preds)
    f1 = f1_score(pred.label_ids, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# arguments for training
training_args = TrainingArguments(
    output_dir="./ticket_results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=5,
    evaluation_strategy="no"
)


# training the model on prepared dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# predict the support ticket into one category
def predict_ticket(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return id2label[torch.argmax(outputs.logits).item()]

predict_ticket("My card was charged but order failed")
