import logging
from typing import Dict, List, Tuple
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SententialRelationExtraction:
    """
    Sentential relation extraction class.

    Attributes:
    - data (pd.DataFrame): Input data.
    - labels (pd.Series): Labels for the data.
    - model (torch.nn.Module): PyTorch model for relation extraction.
    - device (torch.device): Device to run the model on.
    - batch_size (int): Batch size for training.
    - epochs (int): Number of epochs for training.
    - learning_rate (float): Learning rate for the optimizer.
    """

    def __init__(self, data: pd.DataFrame, labels: pd.Series, model: torch.nn.Module, device: torch.device, 
                 batch_size: int = 32, epochs: int = 10, learning_rate: float = 0.001):
        """
        Initialize the SententialRelationExtraction class.

        Args:
        - data (pd.DataFrame): Input data.
        - labels (pd.Series): Labels for the data.
        - model (torch.nn.Module): PyTorch model for relation extraction.
        - device (torch.device): Device to run the model on.
        - batch_size (int): Batch size for training. Defaults to 32.
        - epochs (int): Number of epochs for training. Defaults to 10.
        - learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
        """
        self.data = data
        self.labels = labels
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        """
        Train the model.
        """
        try:
            # Split data into training and validation sets
            train_data, val_data, train_labels, val_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

            # Create data loaders
            train_loader = DataLoader(SententialRelationDataset(train_data, train_labels), batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(SententialRelationDataset(val_data, val_labels), batch_size=self.batch_size, shuffle=False)

            # Set up optimizer and loss function
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            loss_fn = torch.nn.CrossEntropyLoss()

            # Train the model
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                for batch in train_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                logging.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

                # Evaluate the model on the validation set
                self.model.eval()
                total_correct = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = batch
                        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        _, predicted = torch.max(outputs.scores, dim=1)
                        total_correct += (predicted == labels).sum().item()
                accuracy = total_correct / len(val_data)
                logging.info(f'Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}')

        except Exception as e:
            logging.error(f'Training failed: {str(e)}')

    def evaluate(self, test_data: pd.DataFrame, test_labels: pd.Series):
        """
        Evaluate the model on a test set.

        Args:
        - test_data (pd.DataFrame): Test data.
        - test_labels (pd.Series): Test labels.
        """
        try:
            # Create a data loader for the test set
            test_loader = DataLoader(SententialRelationDataset(test_data, test_labels), batch_size=self.batch_size, shuffle=False)

            # Evaluate the model
            self.model.eval()
            total_correct = 0
            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    _, predicted = torch.max(outputs.scores, dim=1)
                    total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / len(test_data)
            logging.info(f'Test Accuracy: {accuracy:.4f}')

        except Exception as e:
            logging.error(f'Evaluation failed: {str(e)}')

class SententialRelationDataset(Dataset):
    """
    Sentential relation extraction dataset class.

    Attributes:
    - data (pd.DataFrame): Input data.
    - labels (pd.Series): Labels for the data.
    """

    def __init__(self, data: pd.DataFrame, labels: pd.Series):
        """
        Initialize the SententialRelationDataset class.

        Args:
        - data (pd.DataFrame): Input data.
        - labels (pd.Series): Labels for the data.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data.iloc[idx, 0])
        attention_mask = torch.tensor(self.data.iloc[idx, 1])
        labels = torch.tensor(self.labels.iloc[idx])
        return input_ids, attention_mask, labels

class CapsuleNetwork(torch.nn.Module):
    """
    Capsule network class.

    Attributes:
    - num_classes (int): Number of classes.
    - num_capsules (int): Number of capsules.
    - num_routes (int): Number of routes.
    """

    def __init__(self, num_classes: int, num_capsules: int, num_routes: int):
        """
        Initialize the CapsuleNetwork class.

        Args:
        - num_classes (int): Number of classes.
        - num_capsules (int): Number of capsules.
        - num_routes (int): Number of routes.
        """
        super(CapsuleNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.capsules = torch.nn.ModuleList([torch.nn.Linear(num_capsules, num_classes) for _ in range(num_routes)])

    def forward(self, input_ids, attention_mask, labels):
        """
        Forward pass.

        Args:
        - input_ids (torch.Tensor): Input IDs.
        - attention_mask (torch.Tensor): Attention mask.
        - labels (torch.Tensor): Labels.

        Returns:
        - outputs (torch.Tensor): Outputs.
        """
        outputs = []
        for capsule in self.capsules:
            output = capsule(input_ids)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs

def main():
    # Load data
    data = pd.read_csv('data.csv')
    labels = pd.read_csv('labels.csv')

    # Create a SententialRelationExtraction instance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CapsuleNetwork(num_classes=8, num_capsules=10, num_routes=3)
    sentential_relation_extraction = SententialRelationExtraction(data, labels, model, device)

    # Train the model
    sentential_relation_extraction.train()

    # Evaluate the model
    test_data = pd.read_csv('test_data.csv')
    test_labels = pd.read_csv('test_labels.csv')
    sentential_relation_extraction.evaluate(test_data, test_labels)

if __name__ == '__main__':
    main()