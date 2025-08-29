import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define constants
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-5
MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset class
class SententialRelationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_seq_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Define a custom trainer class
class SententialRelationTrainer:
    def __init__(self, model: nn.Module, device: torch.device, batch_size: int, epochs: int, lr: float):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_dataloader)

    def evaluate(self, val_dataloader: DataLoader):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, dim=1)
                total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / len(val_dataloader.dataset)
        return accuracy

# Define a custom evaluator class
class SententialRelationEvaluator:
    def __init__(self, model: nn.Module, device: torch.device, batch_size: int):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def evaluate(self, test_dataloader: DataLoader):
        self.model.eval()
        total_correct = 0
        total_pred = []
        total_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_pred.extend(predicted.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())

        accuracy = total_correct / len(test_dataloader.dataset)
        precision = precision_score(total_labels, total_pred, average="macro")
        recall = recall_score(total_labels, total_pred, average="macro")
        f1 = f1_score(total_labels, total_pred, average="macro")

        return accuracy, precision, recall, f1

# Define a main function
def main():
    # Load the dataset
    data = pd.read_csv("data.csv")

    # Split the dataset into training and validation sets
    train_text, val_text, train_labels, val_labels = train_test_split(data["text"], data["label"], random_state=42, test_size=0.2, stratify=data["label"])

    # Create a custom dataset and data loader for training
    train_data = pd.DataFrame({"text": train_text, "label": train_labels})
    val_data = pd.DataFrame({"text": val_text, "label": val_labels})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SententialRelationDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SententialRelationDataset(val_data, tokenizer, MAX_SEQ_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create a model and trainer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)
    model.to(DEVICE)

    trainer = SententialRelationTrainer(model, DEVICE, BATCH_SIZE, EPOCHS, LR)

    # Train the model
    for epoch in range(EPOCHS):
        start_time = time.time()
        loss = trainer.train(train_dataloader)
        end_time = time.time()
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f} seconds")

        # Evaluate the model on the validation set
        accuracy = trainer.evaluate(val_dataloader)
        print(f"Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}")

    # Evaluate the model on the test set
    evaluator = SententialRelationEvaluator(model, DEVICE, BATCH_SIZE)
    test_data = pd.DataFrame({"text": data["text"], "label": data["label"]})
    test_dataset = SententialRelationDataset(test_data, tokenizer, MAX_SEQ_LENGTH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    accuracy, precision, recall, f1 = evaluator.evaluate(test_dataloader)
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()