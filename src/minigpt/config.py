"""Configuration objects for MiniGPT-Train.

为什么不用很复杂的配置系统？
--------------------------------
真实训练平台里经常会用 Hydra、YAML、多层继承配置等工具。它们很强，
但对于第一个训练 infra 项目来说，太容易把注意力从「训练本身」转移走。

这里使用：
1. JSON 文件保存实验配置；
2. dataclass 保存 Python 里的结构化配置；
3. 少量校验帮助你尽早发现拼错的字段。

这样你既能体验「配置驱动训练」，又不会被配置框架本身绕晕。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Dict, Type, TypeVar


T = TypeVar("T")


@dataclass
class ModelConfig:
    """GPT 模型结构配置。

    vocab_size 不写在 JSON 里，因为它取决于 tokenizer 在语料上训练出的词表大小。
    训练启动时会先构建 tokenizer，然后把 vocab_size 填进模型配置。
    """

    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """单卡训练配置。

    这里的字段基本覆盖一个最小预训练系统需要关心的东西：
    数据、输出目录、设备、精度、batch、学习率、checkpoint、日志等。
    """

    data_path: str = "data/tiny_corpus.txt"
    out_dir: str = "runs/tiny"
    device: str = "auto"  # auto / cpu / cuda
    precision: str = "fp32"  # fp32 / fp16 / bf16
    seed: int = 1337

    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000

    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    val_fraction: float = 0.1
    eval_interval: int = 100
    eval_batches: int = 10
    log_interval: int = 10
    checkpoint_interval: int = 500


@dataclass
class ExperimentConfig:
    """完整实验配置：模型配置 + 训练配置。"""

    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _dataclass_from_dict(cls: Type[T], raw: Dict[str, Any]) -> T:
    """把 dict 转成 dataclass，同时检查未知字段。

    这个小检查很实用：如果你在 JSON 里把 learning_rate 写成了 learn_rate，
    没有检查的话训练可能悄悄使用默认学习率；有检查就会直接报错。
    """

    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config field(s) for {cls.__name__}: {unknown}")
    return cls(**raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """从 JSON 文件读取配置。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "model" not in raw or "train" not in raw:
        raise ValueError("Config must contain top-level 'model' and 'train' objects.")

    model_cfg = _dataclass_from_dict(ModelConfig, raw["model"])
    train_cfg = _dataclass_from_dict(TrainConfig, raw["train"])
    return ExperimentConfig(model=model_cfg, train=train_cfg)


def resolve_project_path(project_root: Path, maybe_relative: str | Path) -> Path:
    """把配置里的路径解析成绝对路径。

    JSON 里的路径通常写成 data/tiny_corpus.txt 这种相对路径。
    这样项目挪到别的盘也能跑。真正使用前，我们把它拼到项目根目录下面。
    """

    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return project_root / path
