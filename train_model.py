from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def train_model(train_file, model_path="bert-ad-classifier"):
    """
    Train a BERT model for ad classification.
    """
    # Load dataset
    data = pd.read_csv(train_file)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize data
    inputs = tokenizer(data["AdText"].tolist(), truncation=True, padding=True, return_tensors="pt")
    labels = torch.tensor(data["Category"].astype('category').cat.codes)

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(data["Category"].unique()))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="steps",
        logging_dir="./logs",
        per_device_train_batch_size=8,
        num_train_epochs=3
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=inputs, labels=labels)
    trainer.train()

    # Save the model
    model.save_pretrained(model_path)

if __name__ == "__main__":
    train_model("data/ad_dataset.csv")
