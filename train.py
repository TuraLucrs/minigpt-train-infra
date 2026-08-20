"""Single-device MiniGPT pretraining entry point.

推荐先读这个文件，因为它把整个训练系统串起来了：

1. 读取配置
2. 读取语料
3. 训练 tokenizer
4. 编码文本为 token ids
5. 构造 batcher
6. 创建 MiniGPT 模型
7. 创建 PyTorch AdamW optimizer
8. 选择 fp32/fp16/bf16
9. 执行 forward/loss/backward/optimizer step
10. 记录日志
11. 保存 checkpoint
12. 支持 resume

它是“训练 infra 的第一张地图”。后续改 DDP、FSDP、DeepSpeed 时，
本质上就是替换或扩展这张地图里的某些环节。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Callable, ContextManager

import torch


# 允许用户不安装包，直接在项目根目录运行：python train.py
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.checkpoint import (  # noqa: E402
    build_checkpoint_payload,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from minigpt.config import load_experiment_config, resolve_project_path  # noqa: E402
from minigpt.data import RandomTokenBatcher, split_train_val  # noqa: E402
from minigpt.logging_utils import CSVLogger, memory_stats_mb, reset_peak_memory, synchronize_if_cuda  # noqa: E402
from minigpt.model import (  # noqa: E402
    MiniGPT,
    MiniGPTConfig,
    count_parameters,
    migrate_model_state_dict,
    next_token_cross_entropy,
)
from minigpt.optim import (  # noqa: E402
    cosine_lr,
    load_grad_scaler_state,
    load_optimizer_state,
    set_optimizer_lr,
)
from minigpt.tokenizer import CharTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny GPT model on one device.")
    parser.add_argument("--config", type=str, default="configs/tiny_cpu.json", help="Path to JSON config.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint, for example runs/tiny_cpu/latest.pt.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps from config.")
    parser.add_argument("--device", type=str, default=None, help="Override device: auto / cpu / cuda.")
    parser.add_argument("--precision", type=str, default=None, help="Override precision: fp32 / fp16 / bf16.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--sample", action="store_true", help="Generate a short sample after training.")
    parser.add_argument("--overwrite", action="store_true", help="Allow a fresh run to overwrite old run artifacts.")
    parser.add_argument("--debug_checks", action="store_true", help="Enable extra slow validation checks in loss/data paths.")
    return parser.parse_args()


def apply_cli_overrides(cfg, args: argparse.Namespace) -> None:  # type: ignore[no-untyped-def]
    """把命令行参数覆盖到配置上。

    例如你不想改 JSON，只想临时跑 5 step，可以：
        python train.py --config configs/tiny_cpu.json --max_steps 5
    """

    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.device is not None:
        cfg.train.device = args.device
    if args.precision is not None:
        cfg.train.precision = args.precision
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir


def set_seed(seed: int) -> None:
    """设置随机种子，让实验更容易复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """根据配置选择训练设备。"""

    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warning] config requested cuda, but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def choose_precision(requested: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    """选择训练精度。

    fp16/bf16 只有在 CUDA 上才启用。CPU 上强行混合精度对本项目学习意义不大，
    还容易遇到算子支持差异，所以直接回退 fp32。
    """

    requested = requested.lower()
    if requested not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of: fp32, fp16, bf16")

    if requested == "fp32":
        return "fp32", None

    if device.type != "cuda":
        print(f"[warning] precision={requested} needs CUDA in this project; falling back to fp32.")
        return "fp32", None

    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        print("[warning] bf16 is not supported by this CUDA device; falling back to fp32.")
        return "fp32", None

    dtype = torch.float16 if requested == "fp16" else torch.bfloat16
    return requested, dtype


def make_autocast_context(device: torch.device, amp_dtype: torch.dtype | None) -> Callable[[], ContextManager[None]]:
    """返回一个创建 autocast context 的函数。

    注意这里返回的是“函数”，不是单个 context 对象。因为每个 forward 都应该
    创建一个新的 with context。
    """

    if amp_dtype is None:
        return nullcontext

    def _ctx() -> ContextManager[None]:
        return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)

    return _ctx


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    batcher: RandomTokenBatcher,
    num_batches: int,
    autocast_context: Callable[[], ContextManager[None]],
    debug_checks: bool = False,
) -> float:
    """在验证集上估计 loss。

    训练 loss 只能说明模型在当前随机 batch 上表现如何；val loss 更能反映模型
    是否真的学到了一些可泛化的模式。
    """

    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = batcher.get_batch()
        with autocast_context():
            logits = model(x)
            loss = next_token_cross_entropy(logits, y, debug_checks=debug_checks)
        losses.append(loss.float().item())
    model.train()
    return sum(losses) / len(losses)


def save_training_checkpoint(
    *,
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config_dict: dict,
    best_val_loss: float | None,
    train_batcher: RandomTokenBatcher,
    val_batcher: RandomTokenBatcher,
    tokenizer_hash: str,
    data_hash: str,
    vocab_size: int,
    tokenizer_path: Path,
) -> None:
    """保存完整训练状态。"""

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        config=config_dict,
        best_val_loss=best_val_loss,
    )
    payload["train_batcher_state"] = train_batcher.generator.get_state()
    payload["val_batcher_state"] = val_batcher.generator.get_state()
    payload["tokenizer_hash"] = tokenizer_hash
    payload["data_hash"] = data_hash
    payload["vocab_size"] = vocab_size
    payload["tokenizer_path"] = str(tokenizer_path)
    save_checkpoint(path, payload)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_fingerprint(tokenizer: CharTokenizer) -> str:
    payload = {
        "type": "char",
        "unk_token": tokenizer.unk_token,
        "itos": tokenizer.itos,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def find_resume_tokenizer_path(checkpoint_path: Path, checkpoint: dict, out_dir: Path) -> Path | None:
    candidates: list[Path] = []
    candidates.append(checkpoint_run_dir(checkpoint_path) / "tokenizer.json")

    runtime = checkpoint.get("config", {}).get("runtime", {})
    runtime_tokenizer_path = runtime.get("tokenizer_path") or checkpoint.get("tokenizer_path")
    if runtime_tokenizer_path:
        runtime_path = Path(runtime_tokenizer_path)
        candidates.append(runtime_path if runtime_path.is_absolute() else PROJECT_ROOT / runtime_path)

    candidates.append(out_dir / "tokenizer.json")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def existing_run_artifacts(out_dir: Path) -> list[Path]:
    artifacts: list[Path] = []
    for name in ("train_log.csv", "latest.pt", "tokenizer.json"):
        path = out_dir / name
        if path.exists():
            artifacts.append(path)
    checkpoints_dir = out_dir / "checkpoints"
    if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        artifacts.append(checkpoints_dir)
    return artifacts


def prepare_fresh_out_dir(out_dir: Path, overwrite: bool) -> None:
    artifacts = existing_run_artifacts(out_dir) if out_dir.exists() else []
    if artifacts and not overwrite:
        artifact_list = "\n".join(f"  - {path}" for path in artifacts)
        raise ValueError(
            "Output directory already contains run artifacts. Use --resume to continue it, "
            "choose a new --out_dir, or pass --overwrite to start a fresh run.\n"
            f"{artifact_list}"
        )
    if overwrite:
        for path in artifacts:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def validate_resume_metadata(checkpoint: dict, tokenizer: CharTokenizer, current_tokenizer_hash: str, current_data_hash: str) -> None:
    runtime = checkpoint.get("config", {}).get("runtime", {})

    checkpoint_vocab_size = checkpoint.get("vocab_size", runtime.get("vocab_size"))
    if checkpoint_vocab_size is not None and int(checkpoint_vocab_size) != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size mismatch: checkpoint={checkpoint_vocab_size}, current={tokenizer.vocab_size}"
        )

    checkpoint_tokenizer_hash = checkpoint.get("tokenizer_hash", runtime.get("tokenizer_hash"))
    if checkpoint_tokenizer_hash is not None and checkpoint_tokenizer_hash != current_tokenizer_hash:
        raise ValueError("Tokenizer hash mismatch. Refusing to resume with a different token-id mapping.")

    checkpoint_data_hash = checkpoint.get("data_hash", runtime.get("data_hash"))
    if checkpoint_data_hash is not None and checkpoint_data_hash != current_data_hash:
        print("[warning] data hash differs from the checkpoint. This is allowed, but you are continuing on changed data.")


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    cfg = load_experiment_config(config_path)
    apply_cli_overrides(cfg, args)

    set_seed(cfg.train.seed)

    device = choose_device(cfg.train.device)
    precision_name, amp_dtype = choose_precision(cfg.train.precision, device)
    autocast_context = make_autocast_context(device, amp_dtype)

    data_path = resolve_project_path(PROJECT_ROOT, cfg.train.data_path)
    out_dir = resolve_project_path(PROJECT_ROOT, cfg.train.out_dir)
    resume_path = args.resume.strip()
    if args.overwrite and resume_path:
        raise ValueError("--overwrite is only for fresh runs; do not combine it with --resume.")

    if not resume_path:
        prepare_fresh_out_dir(out_dir, overwrite=args.overwrite)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    text = data_path.read_text(encoding="utf-8")
    data_hash = sha256_text(text)
    tokenizer_path = out_dir / "tokenizer.json"

    checkpoint = None
    checkpoint_path: Path | None = None
    if resume_path:
        checkpoint_path = resolve_project_path(PROJECT_ROOT, resume_path)
        print(f"[resume] loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        source_run_dir = checkpoint_run_dir(checkpoint_path).resolve()
        if out_dir.resolve() != source_run_dir:
            artifacts = existing_run_artifacts(out_dir) if out_dir.exists() else []
            if artifacts:
                artifact_list = "\n".join(f"  - {path}" for path in artifacts)
                raise ValueError(
                    "Resume output directory differs from the checkpoint run directory and already contains "
                    "run artifacts. Choose an empty --out_dir or resume in the original run directory.\n"
                    f"{artifact_list}"
                )
        source_tokenizer_path = find_resume_tokenizer_path(checkpoint_path, checkpoint, out_dir)
        if source_tokenizer_path is None:
            print("[warning] tokenizer.json was not found for the checkpoint; rebuilding from the current data file.")
            tokenizer = CharTokenizer.train_from_text(text)
        else:
            tokenizer = CharTokenizer.load(source_tokenizer_path)
            print(f"[resume] loading tokenizer: {source_tokenizer_path}")
        tokenizer_hash = tokenizer_fingerprint(tokenizer)
        validate_resume_metadata(checkpoint, tokenizer, tokenizer_hash, data_hash)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = CharTokenizer.train_from_text(text)
        tokenizer_hash = tokenizer_fingerprint(tokenizer)
        tokenizer.save(tokenizer_path)

    token_ids = tokenizer.encode(text)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    train_tokens, val_tokens = split_train_val(tokens, cfg.train.val_fraction, cfg.model.block_size)

    train_batcher = RandomTokenBatcher(
        tokens=train_tokens,
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
        device=device,
        seed=cfg.train.seed + 1,
    )
    val_batcher = RandomTokenBatcher(
        tokens=val_tokens,
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
        device=device,
        seed=cfg.train.seed + 2,
    )

    model_config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
    )
    model = MiniGPT(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
        fused=(device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(precision_name == "fp16"),
        init_scale=2.0**12,
    )

    start_step = 0
    best_val_loss: float | None = None
    if checkpoint is not None:
        model.load_state_dict(migrate_model_state_dict(checkpoint["model_state"]))
        load_optimizer_state(optimizer, checkpoint["optimizer_state"])
        load_grad_scaler_state(scaler, checkpoint["scaler_state"])
        restore_rng_state(checkpoint)
        if "train_batcher_state" in checkpoint:
            train_batcher.generator.set_state(checkpoint["train_batcher_state"])
        if "val_batcher_state" in checkpoint:
            val_batcher.generator.set_state(checkpoint["val_batcher_state"])
        start_step = int(checkpoint["step"])
        best_val_loss = checkpoint.get("best_val_loss")

    print("=" * 80)
    print("MiniGPT-Train single-device run")
    print(f"project_root     : {PROJECT_ROOT}")
    print(f"data_path        : {data_path}")
    print(f"out_dir          : {out_dir}")
    print(f"device           : {device}")
    print(f"precision        : {precision_name}")
    print(f"vocab_size       : {tokenizer.vocab_size}")
    print(f"tokenizer_hash   : {tokenizer_hash[:12]}")
    print(f"data_hash        : {data_hash[:12]}")
    print(f"train tokens     : {train_tokens.numel()}")
    print(f"val tokens       : {val_tokens.numel()}")
    print(f"parameters       : {count_parameters(model):,}")
    print(f"start_step       : {start_step}")
    print(f"max_steps        : {cfg.train.max_steps}")
    print("=" * 80)

    log_path = out_dir / "train_log.csv"
    fields = [
        "step",
        "split",
        "loss",
        "lr",
        "tokens_per_sec",
        "gpu_mem_mb",
        "gpu_peak_mb",
        "grad_norm",
        "loss_scale",
        "skipped_step",
    ]

    model.train()
    step = start_step
    config_dict = cfg.to_dict()
    config_dict["runtime"] = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": tokenizer_hash,
        "data_hash": data_hash,
        "precision_used": precision_name,
        "device_used": str(device),
    }

    with CSVLogger(log_path, fields, append=bool(resume_path)) as logger:
        while step < cfg.train.max_steps:
            reset_peak_memory(device)
            synchronize_if_cuda(device)
            step_start = time.perf_counter()

            lr = cosine_lr(
                step=step,
                base_lr=cfg.train.learning_rate,
                min_lr=cfg.train.min_learning_rate,
                warmup_steps=cfg.train.warmup_steps,
                max_steps=cfg.train.max_steps,
            )
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

            # gradient accumulation:
            # 多次 forward/backward 累积梯度，只在最后 optimizer.step()。
            # 等效 global tokens = batch_size * block_size * accumulation_steps。
            accumulated_loss = 0.0
            for _micro_step in range(cfg.train.gradient_accumulation_steps):
                x, y = train_batcher.get_batch()
                with autocast_context():
                    logits = model(x)
                    loss = next_token_cross_entropy(logits, y, debug_checks=args.debug_checks)

                    # 关键点：loss 要除以 accumulation steps。
                    # 否则累积 N 次 backward 后，梯度会比原来大 N 倍。
                    loss_for_backward = loss / cfg.train.gradient_accumulation_steps

                scaler.scale(loss_for_backward).backward()
                accumulated_loss += loss.float().item()

            # fp16 下先恢复真实梯度，再裁剪。GradScaler.step() 会在发现
            # inf/nan 时自动跳过 optimizer.step()。
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip))
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_step = scaler.is_enabled() and scaler.get_scale() < scale_before_step
            optimizer.zero_grad(set_to_none=True)

            synchronize_if_cuda(device)
            step += 1
            elapsed = time.perf_counter() - step_start
            tokens_this_step = cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation_steps
            tokens_per_sec = tokens_this_step / max(elapsed, 1e-9)
            train_loss = accumulated_loss / cfg.train.gradient_accumulation_steps
            gpu_mem_mb, gpu_peak_mb = memory_stats_mb(device)

            if step % cfg.train.log_interval == 0 or step == 1:
                logger.log(
                    {
                        "step": step,
                        "split": "train",
                        "loss": f"{train_loss:.6f}",
                        "lr": f"{lr:.8f}",
                        "tokens_per_sec": f"{tokens_per_sec:.2f}",
                        "gpu_mem_mb": f"{gpu_mem_mb:.2f}",
                        "gpu_peak_mb": f"{gpu_peak_mb:.2f}",
                        "grad_norm": f"{grad_norm:.6f}",
                        "loss_scale": f"{scaler.get_scale():.1f}" if scaler.is_enabled() else "",
                        "skipped_step": str(skipped_step),
                    }
                )
                print(
                    f"step {step:5d} | train loss {train_loss:.4f} | "
                    f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | "
                    f"gpu {gpu_mem_mb:.1f}/{gpu_peak_mb:.1f} MB | "
                    f"grad {grad_norm:.3f}"
                )

            if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
                val_loss = estimate_loss(
                    model,
                    val_batcher,
                    cfg.train.eval_batches,
                    autocast_context,
                    debug_checks=args.debug_checks,
                )
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                logger.log(
                    {
                        "step": step,
                        "split": "val",
                        "loss": f"{val_loss:.6f}",
                        "lr": f"{lr:.8f}",
                        "tokens_per_sec": "",
                        "gpu_mem_mb": f"{gpu_mem_mb:.2f}",
                        "gpu_peak_mb": f"{gpu_peak_mb:.2f}",
                        "grad_norm": "",
                        "loss_scale": f"{scaler.get_scale():.1f}" if scaler.is_enabled() else "",
                        "skipped_step": "",
                    }
                )
                print(f"          | val loss   {val_loss:.4f} | best {best_val_loss:.4f}")

            if step % cfg.train.checkpoint_interval == 0 or step == cfg.train.max_steps:
                numbered_path = out_dir / "checkpoints" / f"step_{step:06d}.pt"
                latest_path = out_dir / "latest.pt"
                save_training_checkpoint(
                    path=numbered_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    config_dict=config_dict,
                    best_val_loss=best_val_loss,
                    train_batcher=train_batcher,
                    val_batcher=val_batcher,
                    tokenizer_hash=tokenizer_hash,
                    data_hash=data_hash,
                    vocab_size=tokenizer.vocab_size,
                    tokenizer_path=tokenizer_path,
                )
                save_training_checkpoint(
                    path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    config_dict=config_dict,
                    best_val_loss=best_val_loss,
                    train_batcher=train_batcher,
                    val_batcher=val_batcher,
                    tokenizer_hash=tokenizer_hash,
                    data_hash=data_hash,
                    vocab_size=tokenizer.vocab_size,
                    tokenizer_path=tokenizer_path,
                )
                print(f"[checkpoint] saved {numbered_path}")
                print(f"[checkpoint] updated {latest_path}")

    if args.sample:
        model.eval()
        prompt = "MiniGPT"
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated = model.generate(prompt_ids, max_new_tokens=80, temperature=0.9)
        print("=" * 80)
        print("Sample:")
        print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
