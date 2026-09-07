"""为 Benchmark 报告补充最小可追溯信息。

这里属于应用层黑箱：输入项目目录和 checkpoint，输出代码版本、文件指纹与启动命令。
它不参与模型计算，但能避免性能数字脱离对应代码和权重后失去复现实验的价值。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Sequence


def sha256_file(path: str | Path) -> str:
    """分块计算文件 SHA-256，避免把大 checkpoint 一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_snapshot(project_root: str | Path) -> dict[str, object]:
    """读取当前 commit、branch 和 dirty 状态；非 Git 目录时返回 unknown。"""

    root = Path(project_root)
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current")
    status = _git_output(root, "status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": None if status is None else bool(status),
    }


def build_provenance(
    project_root: str | Path,
    checkpoint_path: str | Path,
    command: Sequence[str],
) -> dict[str, object]:
    """组装可以直接写进 JSON 报告的最小来源信息。"""

    checkpoint = Path(checkpoint_path).resolve()
    return {
        "git": git_snapshot(project_root),
        "command": list(command),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def build_model_directory_provenance(
    project_root: str | Path,
    model_dir: str | Path,
    command: Sequence[str],
    *,
    hash_weights: bool = False,
) -> dict[str, object]:
    """记录 HF 模型目录；默认不强制扫描几十 GB 权重。"""

    directory = Path(model_dir).resolve()
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json：{directory}")
    weight_files = sorted(directory.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"模型目录缺少 safetensors：{directory}")
    weights = []
    for path in weight_files:
        entry: dict[str, object] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
        }
        if hash_weights:
            entry["sha256"] = sha256_file(path)
        weights.append(entry)
    index_path = directory / "model.safetensors.index.json"
    metadata_names = (
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    )
    metadata_hashes = {
        name: sha256_file(directory / name)
        for name in metadata_names
        if (directory / name).is_file()
    }
    return {
        "git": git_snapshot(project_root),
        "command": list(command),
        "model_dir": str(directory),
        "config_sha256": sha256_file(config_path),
        "metadata_sha256": metadata_hashes,
        "weight_hashes_included": hash_weights,
        "weights": weights,
        "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
    }
