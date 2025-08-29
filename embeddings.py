import logging
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional
from embeddings.config import Config
from embeddings.exceptions import EmbeddingsError
from embeddings.models import EmbeddingModel
from embeddings.utils import get_logger, load_config, validate_config

logger = get_logger(__name__)

class WordEmbeddings:
    def __init__(self, config: Config):
        self.config = config
        self.model = EmbeddingModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def get_word_embeddings(self, words: List[str]) -> np.ndarray:
        """Get word embeddings for a list of words."""
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.detach().numpy()

    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get sentence embeddings for a list of sentences."""
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.detach().numpy()

class SentenceEmbeddings:
    def __init__(self, config: Config):
        self.config = config
        self.model = EmbeddingModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get sentence embeddings for a list of sentences."""
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.detach().numpy()

class EmbeddingModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config.model_name)

    def forward(self, **inputs):
        return self.model(**inputs)

class Embeddings:
    def __init__(self, config: Config):
        self.config = config
        self.word_embeddings = WordEmbeddings(config)
        self.sentence_embeddings = SentenceEmbeddings(config)

    def get_word_embeddings(self, words: List[str]) -> np.ndarray:
        return self.word_embeddings.get_word_embeddings(words)

    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        return self.sentence_embeddings.get_sentence_embeddings(sentences)

def main():
    config = load_config()
    validate_config(config)
    embeddings = Embeddings(config)
    words = ["hello", "world", "this", "is", "a", "test"]
    sentences = ["This is a test sentence.", "Another sentence for testing."]
    word_embeddings = embeddings.get_word_embeddings(words)
    sentence_embeddings = embeddings.get_sentence_embeddings(sentences)
    logger.info(f"Word embeddings: {word_embeddings}")
    logger.info(f"Sentence embeddings: {sentence_embeddings}")

if __name__ == "__main__":
    main()