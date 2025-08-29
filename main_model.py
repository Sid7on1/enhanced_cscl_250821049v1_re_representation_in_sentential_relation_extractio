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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SententialRelationExtractionModel(nn.Module):
    """
    Sentential relation extraction model using dynamic routing in capsules.

    Attributes:
    ----------
    num_classes : int
        Number of relation classes.
    num_capsules : int
        Number of capsules in the model.
    num_routes : int
        Number of routing iterations.
    embedding_dim : int
        Dimension of the word embeddings.
    """

    def __init__(self, num_classes: int, num_capsules: int, num_routes: int, embedding_dim: int):
        super(SententialRelationExtractionModel, self).__init__()
        self.num_classes = num_classes
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.embedding_dim = embedding_dim
        self.capsules = nn.ModuleList([Capsule(embedding_dim, num_capsules) for _ in range(num_routes)])
        self.fc = nn.Linear(num_capsules * embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Output tensor.
        """
        for capsule in self.capsules:
            x = capsule(x)
        x = x.view(-1, self.num_capsules * self.embedding_dim)
        x = self.fc(x)
        return x


class Capsule(nn.Module):
    """
    Capsule module.

    Attributes:
    ----------
    embedding_dim : int
        Dimension of the word embeddings.
    num_capsules : int
        Number of capsules in the module.
    """

    def __init__(self, embedding_dim: int, num_capsules: int):
        super(Capsule, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_capsules = num_capsules
        self.weights = nn.Parameter(torch.randn(embedding_dim, num_capsules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the capsule.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Output tensor.
        """
        x = torch.matmul(x, self.weights)
        x = torch.sigmoid(x)
        return x


class SententialRelationExtractionDataset(Dataset):
    """
    Sentential relation extraction dataset.

    Attributes:
    ----------
    data : List[Tuple[str, str, int]]
        List of tuples containing the sentence, relation, and label.
    """

    def __init__(self, data: List[Tuple[str, str, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the item at the specified index.

        Parameters:
        ----------
        index : int
            Index of the item.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the input tensor and the label tensor.
        """
        sentence, relation, label = self.data[index]
        input_tensor = torch.tensor([ord(c) for c in sentence], dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_tensor, label_tensor


class SententialRelationExtractionTrainer:
    """
    Sentential relation extraction trainer.

    Attributes:
    ----------
    model : SententialRelationExtractionModel
        Sentential relation extraction model.
    device : torch.device
        Device to use for training.
    optimizer : torch.optim.Optimizer
        Optimizer to use for training.
    criterion : torch.nn.CrossEntropyLoss
        Loss function to use for training.
    """

    def __init__(self, model: SententialRelationExtractionModel, device: torch.device, optimizer: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, dataset: SententialRelationExtractionDataset, batch_size: int, epochs: int) -> None:
        """
        Train the model on the specified dataset.

        Parameters:
        ----------
        dataset : SententialRelationExtractionDataset
            Dataset to train on.
        batch_size : int
            Batch size to use for training.
        epochs : int
            Number of epochs to train for.
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in data_loader:
                input_tensor, label_tensor = batch
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input_tensor)
                loss = self.criterion(output, label_tensor)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    def evaluate(self, dataset: SententialRelationExtractionDataset, batch_size: int) -> Dict[str, float]:
        """
        Evaluate the model on the specified dataset.

        Parameters:
        ----------
        dataset : SententialRelationExtractionDataset
            Dataset to evaluate on.
        batch_size : int
            Batch size to use for evaluation.

        Returns:
        -------
        Dict[str, float]
            Dictionary containing the accuracy, precision, recall, and F1 score.
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        total_correct = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        with torch.no_grad():
            for batch in data_loader:
                input_tensor, label_tensor = batch
                input_tensor, label_tensor = input_tensor.to(self.device), label_tensor.to(self.device)
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == label_tensor).sum().item()
                total_precision += precision_score(label_tensor.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                total_recall += recall_score(label_tensor.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                total_f1 += f1_score(label_tensor.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        accuracy = total_correct / len(dataset)
        precision = total_precision / len(data_loader)
        recall = total_recall / len(data_loader)
        f1 = total_f1 / len(data_loader)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def main() -> None:
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the model
    num_classes = 8
    num_capsules = 10
    num_routes = 3
    embedding_dim = 128
    model = SententialRelationExtractionModel(num_classes, num_capsules, num_routes, embedding_dim)

    # Set up the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Set up the trainer
    trainer = SententialRelationExtractionTrainer(model, device, optimizer, criterion)

    # Load the dataset
    data = pd.read_csv('data.csv')
    dataset = SententialRelationExtractionDataset([(row['sentence'], row['relation'], row['label']) for index, row in data.iterrows()])

    # Train the model
    batch_size = 32
    epochs = 10
    trainer.train(dataset, batch_size, epochs)

    # Evaluate the model
    evaluation_results = trainer.evaluate(dataset, batch_size)
    logger.info(f'Accuracy: {evaluation_results["accuracy"]}')
    logger.info(f'Precision: {evaluation_results["precision"]}')
    logger.info(f'Recall: {evaluation_results["recall"]}')
    logger.info(f'F1: {evaluation_results["f1"]}')


if __name__ == '__main__':
    main()