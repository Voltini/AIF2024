from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
import pandas as pd

import torch
from torch.utils.data import Dataset


class NlpDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data.to_list()
        self.labels = labels.tolist()
        self.encodings = tokenizer(self.data, truncation=True, padding=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # Load the datasets
    train_df = pd.read_csv("./data/train.csv", names=["label", "title", "text"]).sample(
        40000
    )
    test_df = pd.read_csv("./data/test.csv", names=["label", "title", "text"]).sample(
        2000
    )
    train_text, train_labels = train_df["text"], train_df["label"] - 1
    test_text, test_labels = test_df["text"], test_df["label"] - 1

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = NlpDataset(train_text, train_labels, tokenizer)
    test_dataset = NlpDataset(test_text, test_labels, tokenizer)

    unique_labels = train_df["label"].unique()

    # Load a pre-trained BERT model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(unique_labels)
    )

    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        num_train_epochs=3,  # Total number of training epochs
        per_device_train_batch_size=16,  # Batch size per device during training
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Fine-tune the model
    trainer.train()

    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_tokenizer")
