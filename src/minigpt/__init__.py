"""MiniGPT-Train learning package.

这个包故意很小：你可以从 tokenizer -> data -> model -> optim -> train
一路读下来，看到一个 GPT 预训练系统最核心的骨架。
"""

from .config import ExperimentConfig, ModelConfig, TrainConfig
from .model import MiniGPT, MiniGPTConfig
from .tokenizer import CharTokenizer

__all__ = [
    "CharTokenizer",
    "ExperimentConfig",
    "MiniGPT",
    "MiniGPTConfig",
    "ModelConfig",
    "TrainConfig",
]
