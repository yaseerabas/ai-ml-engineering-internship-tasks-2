from transformers import pipeline
import json

# load the dataset from the json file
with open("example_ticket_support_dataset.json", "r") as f:
    data = json.load(f)

tickets = data["tickets"]
categories = data["categories"]

zero_shot_pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# predict the tickets
def zero_shot_predict(text):
    result = zero_shot_pipe(text, categories)
    return result["labels"][0]


for t in tickets:
    print(t, "→", zero_shot_predict(t))