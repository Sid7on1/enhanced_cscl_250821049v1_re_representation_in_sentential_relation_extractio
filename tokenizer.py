# tokenizer.py
"""
Text tokenization utilities.
"""

import logging
import re
import string
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
DEFAULT_TOKENIZER_CONFIG = {
    "tokenizer_type": "wordpiece",
    "max_length": 512,
    "min_length": 2,
    "vocab_size": 50000,
}

class TokenizerType(Enum):
    """Tokenizer types."""
    WORDPIECE = "wordpiece"
    BPE = "bpe"
    WORD = "word"

class TokenizerConfig:
    """Tokenizer configuration."""
    def __init__(self, config_file: str = TOKENIZER_CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load tokenizer configuration from file."""
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Tokenizer config file not found: {self.config_file}")
            return DEFAULT_TOKENIZER_CONFIG

    def save_config(self):
        """Save tokenizer configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

class Tokenizer(ABC):
    """Abstract base class for tokenizers."""
    def __init__(self, config: TokenizerConfig):
        self.config = config

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        pass

class WordPieceTokenizer(Tokenizer):
    """WordPiece tokenizer."""
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self.vocab = self.load_vocab()

    def load_vocab(self) -> Dict:
        """Load vocabulary from file."""
        vocab_file = f"{self.config['vocab_size']}_vocab.txt"
        try:
            with open(vocab_file, "r") as f:
                return {line.strip(): i for i, line in enumerate(f)}
        except FileNotFoundError:
            logger.warning(f"Vocabulary file not found: {vocab_file}")
            return {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using WordPiece algorithm."""
        tokens = []
        for word in text.split():
            word = re.sub(r"[^a-zA-Z0-9]", "", word)
            if word:
                subwords = self.wordpiece_tokenize(word)
                tokens.extend(subwords)
        return tokens

    def wordpiece_tokenize(self, word: str) -> List[str]:
        """Tokenize word using WordPiece algorithm."""
        subwords = []
        while word:
            best_subword = self.find_best_subword(word)
            subwords.append(best_subword)
            word = word[len(best_subword):]
        return subwords

    def find_best_subword(self, word: str) -> str:
        """Find best subword for word."""
        best_subword = ""
        best_score = 0
        for subword in self.vocab:
            score = self.score_subword(word, subword)
            if score > best_score:
                best_subword = subword
                best_score = score
        return best_subword

    def score_subword(self, word: str, subword: str) -> float:
        """Score subword for word."""
        score = 0
        for i in range(len(word)):
            if word[i:i+len(subword)] == subword:
                score += 1
        return score / len(word)

class BPETokenizer(Tokenizer):
    """BPE tokenizer."""
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self.vocab = self.load_vocab()

    def load_vocab(self) -> Dict:
        """Load vocabulary from file."""
        vocab_file = f"{self.config['vocab_size']}_vocab.txt"
        try:
            with open(vocab_file, "r") as f:
                return {line.strip(): i for i, line in enumerate(f)}
        except FileNotFoundError:
            logger.warning(f"Vocabulary file not found: {vocab_file}")
            return {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE algorithm."""
        tokens = []
        for word in text.split():
            word = re.sub(r"[^a-zA-Z0-9]", "", word)
            if word:
                subwords = self.bpe_tokenize(word)
                tokens.extend(subwords)
        return tokens

    def bpe_tokenize(self, word: str) -> List[str]:
        """Tokenize word using BPE algorithm."""
        subwords = []
        while word:
            best_subword = self.find_best_subword(word)
            subwords.append(best_subword)
            word = word[len(best_subword):]
        return subwords

    def find_best_subword(self, word: str) -> str:
        """Find best subword for word."""
        best_subword = ""
        best_score = 0
        for subword in self.vocab:
            score = self.score_subword(word, subword)
            if score > best_score:
                best_subword = subword
                best_score = score
        return best_subword

    def score_subword(self, word: str, subword: str) -> float:
        """Score subword for word."""
        score = 0
        for i in range(len(word)):
            if word[i:i+len(subword)] == subword:
                score += 1
        return score / len(word)

class WordTokenizer(Tokenizer):
    """Word tokenizer."""
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using word tokenizer."""
        return text.split()

def main():
    config = TokenizerConfig()
    tokenizer = WordPieceTokenizer(config)
    text = "This is an example sentence."
    tokens = tokenizer.tokenize(text)
    print(tokens)

if __name__ == "__main__":
    main()