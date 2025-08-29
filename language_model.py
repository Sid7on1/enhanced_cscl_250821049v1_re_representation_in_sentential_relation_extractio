import logging
import torch
import numpy as np
from typing import Dict, List, Tuple
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
import pandas as pd
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

class SentimentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        text = self.data.iloc[item, 0]
        label = self.data.iloc[item, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

class SentimentTrainer:
    def __init__(self, model: SentimentModel, device: torch.device, optimizer: Adam, scheduler: torch.optim.lr_scheduler.StepLR):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, train_dataloader: DataLoader, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            self.scheduler.step()
            logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

    def evaluate(self, val_dataloader: DataLoader):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.scores, dim=1)
                total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / len(val_dataloader.dataset)
        logger.info(f"Validation Accuracy: {accuracy:.4f}")

def load_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])
    return data

def train_model(model: SentimentModel, device: torch.device, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int):
    trainer = SentimentTrainer(model, device, Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), torch.optim.lr_scheduler.StepLR(Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), step_size=1, gamma=0.1))
    trainer.train(train_dataloader, epochs)
    trainer.evaluate(val_dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_path = "data.csv"
    data = load_data(data_path)
    data = preprocess_data(data)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SentimentDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SentimentDataset(val_data, tokenizer, MAX_SEQ_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentModel()
    model.to(device)

    train_model(model, device, train_dataloader, val_dataloader, EPOCHS)

if __name__ == "__main__":
    main()