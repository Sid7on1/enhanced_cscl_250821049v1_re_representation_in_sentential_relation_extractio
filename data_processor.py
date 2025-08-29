import logging
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
DATA_DIR = 'data'
TOKENIZER_NAME = 'bert-base-uncased'
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32

# Data structure for text data
@dataclass
class TextData:
    text: str
    label: str

# Enum for data labels
class Label(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'

# Abstract base class for data processors
class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: List[TextData]) -> List[TextData]:
        pass

# Concrete data processor for text data
class TextDataProcessor(DataProcessor):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def process(self, data: List[TextData]) -> List[TextData]:
        processed_data = []
        for text_data in data:
            inputs = self.tokenizer(text_data.text, return_tensors='pt', max_length=MAX_SEQ_LENGTH, padding='max_length', truncation=True)
            processed_data.append(TextData(text_data.text, text_data.label, **inputs))
        return processed_data

# Data loader for text data
class TextDataset(Dataset):
    def __init__(self, data: List[TextData], tokenizer: AutoTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_data = self.data[idx]
        inputs = self.tokenizer(text_data.text, return_tensors='pt', max_length=MAX_SEQ_LENGTH, padding='max_length', truncation=True)
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': text_data.label
        }

# Data loader for text data
class TextDataLoader:
    def __init__(self, data: List[TextData], tokenizer: AutoTokenizer, batch_size: int):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __iter__(self):
        dataset = TextDataset(self.data, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return iter(data_loader)

# Configuration class for data processor
@dataclass
class DataProcessorConfig:
    data_dir: str
    tokenizer_name: str
    max_seq_length: int
    batch_size: int

# Main class for data processor
class DataProcessorMain:
    def __init__(self, config: DataProcessorConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.data_processor = TextDataProcessor(self.tokenizer)

    def process_data(self, data: List[TextData]) -> List[TextData]:
        processed_data = self.data_processor.process(data)
        return processed_data

    def load_data(self, data_dir: str) -> List[TextData]:
        data_files = os.listdir(data_dir)
        data = []
        for file in data_files:
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r') as f:
                for line in f:
                    text, label = line.strip().split('\t')
                    data.append(TextData(text, label))
        return data

    def save_data(self, data: List[TextData], output_dir: str):
        output_file = os.path.join(output_dir, 'processed_data.json')
        with open(output_file, 'w') as f:
            json.dump([{'text': text_data.text, 'label': text_data.label} for text_data in data], f)

    def run(self):
        data_dir = self.config.data_dir
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        data = self.load_data(data_dir)
        processed_data = self.process_data(data)
        self.save_data(processed_data, output_dir)

# Main function
def main():
    config = DataProcessorConfig(
        data_dir='data',
        tokenizer_name=TOKENIZER_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE
    )
    data_processor_main = DataProcessorMain(config)
    data_processor_main.run()

if __name__ == '__main__':
    main()