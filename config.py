import os
import logging
from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path

import numpy as np
import torch

from enhanced_cs.utils.logging import setup_logging
from enhanced_cs.utils.validation import validate_config

logger = logging.getLogger(__name__)

CONFIG_DEFAULTS = {
    "model": {
        "type": "CapsuleNetwork",
        "hidden_dim": 512,
        "num_capsules": 16,
        "num_routing_iterations": 3,
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    "data": {
        "dataset": "Tacred",
        "data_path": "data/tacred/",
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4,
        "validation_split": 0.1,
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs": 50,
        "checkpoint_dir": "checkpoints/",
        "checkpoint_interval": 5,
        "early_stopping_patience": 10,
        "gradient_clipping": 1.0,
    },
}

MODEL_TYPES = ["CapsuleNetwork", "TransformerModel", "LSTMModel"]
DATASETS = ["Tacred", "TacredRev", "Retacred", "Conll04", "Wikidata"]

CONFIG_SCHEMA = {
    "type": str,
    "properties": {
        "model": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": MODEL_TYPES,
                },
                "hidden_dim": {"type": "integer", "minimum": 64},
                "num_capsules": {"type": "integer", "minimum": 1},
                "num_routing_iterations": {"type": "integer", "minimum": 1},
                "dropout": {"type": "number", "minimum": 0, "maximum": 1},
                "learning_rate": {"type": "number", "minimum": 0},
            },
            "required": ["type", "hidden_dim", "num_capsules", "num_routing_iterations", "dropout", "learning_rate"],
        },
        "data": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "enum": DATASETS,
                },
                "data_path": {"type": "string"},
                "batch_size": {"type": "integer", "minimum": 1},
                "shuffle": {"type": "boolean"},
                "num_workers": {"type": "integer", "minimum": 0},
                "validation_split": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["dataset", "data_path", "batch_size", "shuffle", "num_workers", "validation_split"],
        },
        "training": {
            "type": "object",
            "properties": {
                "device": {"type": "string"},
                "num_epochs": {"type": "integer", "minimum": 1},
                "checkpoint_dir": {"type": "string"},
                "checkpoint_interval": {"type": "integer", "minimum": 1},
                "early_stopping_patience": {"type": "integer", "minimum": 0},
                "gradient_clipping": {"type": "number", "minimum": 0},
            },
            "required": ["device", "num_epochs", "checkpoint_dir", "checkpoint_interval", "early_stopping_patience", "gradient_clipping"],
        },
    },
    "required": ["model", "data", "training"],
}


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.config_defaults = CONFIG_DEFAULTS.copy()
        self._validate_config()
        self._setup_directories()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path:
            return self.config_defaults

        config_path = Path(self.config_path)
        if not config_path.exists():
            logger.warning(f"Config file '{config_path}' not found. Using defaults.")
            return self.config_defaults

        with open(config_path, "r") as f:
            try:
                loaded_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Error loading config file '{config_path}': {e}")
                raise

        # Merge loaded config with defaults
        merged_config = {**self.config_defaults, **loaded_config}

        return merged_config

    def _validate_config(self):
        validate_config(self.config, CONFIG_SCHEMA)

    def _setup_directories(self):
        dirs_to_create = [self.config["training"]["checkpoint_dir"]]
        for dir_path in dirs_to_create:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")

    @property
    def model(self) -> Dict[str, Any]:
        return self.config["model"]

    @property
    def data(self) -> Dict[str, Any]:
        return self.config["data"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.config["training"]

    def to_dict(self) -> Dict[str, Any]:
        return self.config

    def to_json(self, json_path: str):
        json_path = Path(json_path)
        with open(json_path, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to: {json_path}")

    def to_yaml(self, yaml_path: str):
        yaml_path = Path(yaml_path)
        with open(yaml_path, "w") as f:
            yaml.dump(self.config, f)
        logger.info(f"Saved config to: {yaml_path}")

    def update_from_dict(self, update_dict: Dict[str, Any]):
        self.config.update(update_dict)
        self._validate_config()

    def update_from_file(self, config_file: str):
        config_file = Path(config_file)
        with open(config_file, "r") as f:
            update_dict = yaml.safe_load(f)
        self.update_from_dict(update_dict)


class ConfigError(Exception):
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the configuration file.

    Returns:
    - Dict[str, Any]: Loaded configuration.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to a YAML file.

    Parameters:
    - config (Dict[str, Any]): Configuration to save.
    - output_path (str): Path to save the configuration file.
    """
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to: {output_path}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]):
    """
    Update configuration with new values.

    Parameters:
    - config (Dict[str, Any]): Original configuration.
    - updates (Dict[str, Any]): Updates to apply to the configuration.

    Returns:
    - Dict[str, Any]: Updated configuration.
    """
    updated_config = {**config, **updates}
    return updated_config


def setup_seed(seed: int):
    """
    Set random seed for reproducibility.

    Parameters:
    - seed (int): Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")


def setup_environment(config: Dict[str, Any]):
    """
    Setup environment based on configuration.

    Parameters:
    - config (Dict[str, Any]): Configuration containing environment settings.
    """
    # Set random seed for reproducibility
    setup_seed(config["training"]["random_seed"])

    # Set PyTorch device
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA device requested but not available. Using CPU instead.")
        device = "cpu"
    config["training"]["device"] = device
    logger.info(f"Using device: {device}")


def setup_logging_from_config(config: Dict[str, Any]):
    """
    Setup logging based on configuration.

    Parameters:
    - config (Dict[str, Any]): Configuration containing logging settings.
    """
    log_level = config["logging"]["level"]
    log_file = config["logging"]["file"]

    # Setup logging with specified settings
    setup_logging(level=log_level, file=log_file)


def setup_project(config_path: Optional[str] = None):
    """
    Setup the project environment based on configuration.

    Parameters:
    - config_path (str, optional): Path to the configuration file. Defaults to None.
    """
    # Load configuration
    config = Config(config_path)

    # Setup logging based on config
    setup_logging_from_config(config.to_dict())

    # Setup environment based on config
    setup_environment(config.to_dict())

    return config


def main():
    config_path = "path/to/config.yaml"
    config = setup_project(config_path)

    # Example usage of the config
    model_type = config.model["type"]
    learning_rate = config.model["learning_rate"]
    dataset = config.data["dataset"]
    batch_size = config.data["batch_size"]
    num_epochs = config.training["num_epochs"]

    logger.info(f"Model Type: {model_type}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Num Epochs: {num_epochs}")


if __name__ == "__main__":
    main()