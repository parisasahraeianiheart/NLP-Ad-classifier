from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-ad-classifier")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    inputs = tokenizer(content["text"], return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
