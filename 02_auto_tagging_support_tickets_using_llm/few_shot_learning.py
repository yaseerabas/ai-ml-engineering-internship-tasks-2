from transformers import pipeline
import json

# load dataset from json file
with open("example_ticket_support_dataset.json", "r") as f:
    data = json.load(f)

tickets = data["tickets"]
categories = data["categories"]

# initialize pipeline
pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=10
)

# make predictions
def few_shot_prompt(ticket):
    prompt = f"""
Classify the support ticket into one category:
Billing, Technical Issue, Account, Feature Request

Examples:
"I was charged twice" -> Billing
"App is not opening" -> Technical Issue
"Can't login to my account" -> Account
"Please add dark mode" -> Feature Request

Ticket: "{ticket}"
Category:
"""
    return pipe(prompt)[0]["generated_text"]

for t in tickets:
    print(few_shot_prompt(t))