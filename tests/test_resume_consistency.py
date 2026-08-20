"""Verify that uninterrupted and checkpoint-resumed training are identical.

Run:
    python tests/test_resume_consistency.py
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def assert_nested_equal(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"{path}: tensor mismatch")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"{path}: dictionary mismatch")
        for key in left:
            assert_nested_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"{path}: sequence mismatch")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_nested_equal(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def run_train(config_path: Path, out_dir: Path, resume: Path | None = None) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--config",
        str(config_path),
        "--out_dir",
        str(out_dir),
        "--device",
        "cpu",
    ]
    if resume is None:
        command.append("--overwrite")
    else:
        command.extend(["--resume", str(resume)])

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Training subprocess failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        continuous_dir = work_dir / "continuous"
        resumed_dir = work_dir / "resumed"
        config_path = work_dir / "resume_test.json"

        config = {
            "model": {
                "block_size": 8,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
                "dropout": 0.1,
            },
            "train": {
                "data_path": str(PROJECT_ROOT / "data" / "tiny_corpus.txt"),
                "out_dir": str(continuous_dir),
                "device": "cpu",
                "precision": "fp32",
                "seed": 1337,
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "max_steps": 4,
                "learning_rate": 0.001,
                "min_learning_rate": 0.0001,
                "warmup_steps": 1,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.95,
                "adam_eps": 1e-8,
                "grad_clip": 1.0,
                "val_fraction": 0.1,
                "eval_interval": 2,
                "eval_batches": 2,
                "log_interval": 1,
                "checkpoint_interval": 2,
            },
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        run_train(config_path, continuous_dir)

        source_checkpoint = continuous_dir / "checkpoints" / "step_000002.pt"
        resume_checkpoint = resumed_dir / "checkpoints" / "step_000002.pt"
        resume_checkpoint.parent.mkdir(parents=True)
        shutil.copy2(source_checkpoint, resume_checkpoint)
        shutil.copy2(continuous_dir / "tokenizer.json", resumed_dir / "tokenizer.json")

        run_train(config_path, resumed_dir, resume=resume_checkpoint)

        continuous = torch.load(continuous_dir / "latest.pt", map_location="cpu")
        resumed = torch.load(resumed_dir / "latest.pt", map_location="cpu")

        # Runtime output paths differ, so compare only trajectory-defining state.
        keys = (
            "model_state",
            "optimizer_state",
            "scaler_state",
            "step",
            "best_val_loss",
            "rng_state",
            "train_batcher_state",
            "val_batcher_state",
            "tokenizer_hash",
            "data_hash",
            "vocab_size",
        )
        for key in keys:
            assert_nested_equal(continuous[key], resumed[key], key)

    print("Exact resume consistency test passed.")


if __name__ == "__main__":
    main()
