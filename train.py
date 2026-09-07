"""MiniGPT 单设备预训练入口。

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

它是“训练 infra 的第一张地图”。从 v0.3 开始，这条训练路径作为知识基础和回归 workload
保留；项目主线转入独立推理引擎。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch


# 允许用户不安装包，直接在项目根目录运行：python train.py
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.checkpoint import (  # noqa: E402
    build_checkpoint_payload,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    update_latest_checkpoint,
)
from minigpt.config import load_experiment_config, resolve_project_path  # noqa: E402
from minigpt.data import RandomTokenBatcher, split_train_val  # noqa: E402
from minigpt.inference import GenerationConfig, InferenceEngine, MiniGPTModelRunner  # noqa: E402
from minigpt.logging_utils import CSVLogger  # noqa: E402
from minigpt.model import (  # noqa: E402
    MiniGPT,
    MiniGPTConfig,
    count_parameters,
    migrate_model_state_dict,
    next_token_cross_entropy,
)
from minigpt.optim import (  # noqa: E402
    build_adamw_param_groups,
    cosine_lr,
    load_grad_scaler_state,
    load_optimizer_state,
    set_optimizer_lr,
)
from minigpt.runtime import DeviceIntervalTimer, RuntimeContext  # noqa: E402
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


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    batcher: RandomTokenBatcher,
    num_batches: int,
    runtime: RuntimeContext,
    debug_checks: bool = False,
) -> float:
    """在验证集上估计 loss。

    训练 loss 只能说明模型在当前随机 batch 上表现如何；val loss 更能反映模型
    是否真的学到了一些可泛化的模式。
    """

    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
    for _ in range(num_batches):
        x, y = batcher.get_batch()
        with runtime.autocast():
            logits = model(x)
            loss = next_token_cross_entropy(logits, y, debug_checks=debug_checks)
        loss_sum += loss.detach().float()
    model.train()
    return (loss_sum / num_batches).item()


def build_training_checkpoint_payload(
    *,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    optimizer_step: int,
    config_dict: dict,
    best_val_loss: float | None,
    train_batcher: RandomTokenBatcher,
    val_batcher: RandomTokenBatcher,
    tokenizer_hash: str,
    data_hash: str,
    vocab_size: int,
    tokenizer_path: Path,
) -> dict:
    """在写文件前一次性组装完整训练状态。"""

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        config=config_dict,
        best_val_loss=best_val_loss,
    )
    payload["iteration_step"] = step
    payload["optimizer_step"] = optimizer_step
    payload["train_batcher_state"] = train_batcher.generator.get_state()
    payload["val_batcher_state"] = val_batcher.generator.get_state()
    payload["tokenizer_hash"] = tokenizer_hash
    payload["data_hash"] = data_hash
    payload["vocab_size"] = vocab_size
    payload["tokenizer_path"] = str(tokenizer_path)
    return payload


def optimizer_completed_steps(optimizer: torch.optim.Optimizer) -> int:
    """读取 AdamW 已成功完成的参数更新次数。

    当前 GPT 的所有可训练参数都会参与每轮 backward，因此读取第一个已有状态的
    参数即可。这个读取只发生在统计窗口边界，避免每个 step 都把设备标量同步回 CPU。
    """

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state or "step" not in state:
                continue
            completed = state["step"]
            return int(completed.item()) if isinstance(completed, torch.Tensor) else int(completed)
    return 0


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

    runtime = RuntimeContext.create(cfg.train.device, cfg.train.precision)
    device = runtime.device
    precision_name = runtime.precision
    random.seed(cfg.train.seed)
    runtime.manual_seed(cfg.train.seed)

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
        # checkpoint 统一先落到 CPU：模型和 optimizer 状态随后按各自目标设备恢复，
        # CPU RNG 与 batcher Generator 状态则不会被误映射到 CUDA。
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
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
        build_adamw_param_groups(model, cfg.train.weight_decay),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        eps=cfg.train.adam_eps,
        fused=runtime.capabilities.supports_fused_adamw,
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
        load_optimizer_state(optimizer, checkpoint["optimizer_state"], model)
        load_grad_scaler_state(scaler, checkpoint["scaler_state"])
        restore_rng_state(checkpoint)
        if "train_batcher_state" in checkpoint:
            train_batcher.generator.set_state(checkpoint["train_batcher_state"].cpu())
        if "val_batcher_state" in checkpoint:
            val_batcher.generator.set_state(checkpoint["val_batcher_state"].cpu())
        start_step = int(checkpoint.get("iteration_step", checkpoint["step"]))
        best_val_loss = checkpoint.get("best_val_loss")

    start_optimizer_step = optimizer_completed_steps(optimizer)

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
    print(f"optimizer_step   : {start_optimizer_step}")
    print(f"max_steps        : {cfg.train.max_steps}")
    print("=" * 80)

    log_path = out_dir / "train_log.csv"
    fields = [
        "step",
        "optimizer_step",
        "split",
        "loss",
        "lr",
        "tokens_per_sec",
        "gpu_mem_mb",
        "gpu_peak_mb",
        "grad_norm_last",
        "grad_norm_max",
        "loss_scale",
        "skipped_steps",
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

    interval_timer = DeviceIntervalTimer(runtime)
    window_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    window_micro_batches = 0
    window_tokens = 0
    window_steps = 0
    window_optimizer_step_start = start_optimizer_step
    window_grad_norm_last = torch.zeros((), dtype=torch.float32, device=device)
    window_grad_norm_max = torch.zeros((), dtype=torch.float32, device=device)
    runtime.reset_peak_memory()
    interval_timer.start()

    with CSVLogger(log_path, fields, append=bool(resume_path)) as logger:
        while step < cfg.train.max_steps:
            next_step = step + 1
            should_log = next_step % cfg.train.log_interval == 0 or next_step == 1
            should_eval = next_step % cfg.train.eval_interval == 0 or next_step == cfg.train.max_steps
            should_checkpoint = (
                next_step % cfg.train.checkpoint_interval == 0 or next_step == cfg.train.max_steps
            )
            # 任一慢速旁路都先关闭纯训练计时窗口，避免 eval/checkpoint I/O
            # 被错误计入训练 step 时间。
            should_close_window = should_log or should_eval or should_checkpoint

            lr = cosine_lr(
                step=step,
                base_lr=cfg.train.learning_rate,
                min_lr=cfg.train.min_learning_rate,
                warmup_steps=cfg.train.warmup_steps,
                max_steps=cfg.train.max_steps,
            )
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

            # 梯度累积：多次 forward/backward 累积梯度，只在最后 optimizer.step()。
            # 等效 global tokens = batch_size * block_size * accumulation_steps。
            for _micro_step in range(cfg.train.gradient_accumulation_steps):
                x, y = train_batcher.get_batch()
                with runtime.autocast():
                    logits = model(x)
                    loss = next_token_cross_entropy(logits, y, debug_checks=args.debug_checks)

                    # 关键点：loss 要除以 accumulation steps。
                    # 否则累积 N 次 backward 后，梯度会比原来大 N 倍。
                    loss_for_backward = loss / cfg.train.gradient_accumulation_steps

                scaler.scale(loss_for_backward).backward()
                window_loss_sum += loss.detach().float()
                window_micro_batches += 1

            # fp16 下先恢复真实梯度，再裁剪。GradScaler.step() 会在发现
            # inf/nan 时自动跳过 optimizer.step()。
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            grad_norm_detached = grad_norm_tensor.detach().float()
            window_grad_norm_last = grad_norm_detached
            window_grad_norm_max = torch.maximum(window_grad_norm_max, grad_norm_detached)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            tokens_this_step = cfg.train.batch_size * cfg.model.block_size * cfg.train.gradient_accumulation_steps
            window_tokens += tokens_this_step
            window_steps += 1

            if should_close_window:
                elapsed = interval_timer.elapsed_seconds()
                tokens_per_sec = window_tokens / max(elapsed, 1e-9)
                train_loss = (window_loss_sum / window_micro_batches).item()
                grad_norm_last = window_grad_norm_last.item()
                grad_norm_max = window_grad_norm_max.item()
                scale_after_step = scaler.get_scale() if scaler.is_enabled() else 1.0
                optimizer_step = optimizer_completed_steps(optimizer)
                successful_steps = optimizer_step - window_optimizer_step_start
                skipped_steps = max(0, window_steps - successful_steps)
                gpu_mem_mb, gpu_peak_mb = runtime.memory_stats_mb()
                logger.log(
                    {
                        "step": step,
                        "optimizer_step": optimizer_step,
                        "split": "train",
                        "loss": f"{train_loss:.6f}",
                        "lr": f"{lr:.8f}",
                        "tokens_per_sec": f"{tokens_per_sec:.2f}",
                        "gpu_mem_mb": f"{gpu_mem_mb:.2f}",
                        "gpu_peak_mb": f"{gpu_peak_mb:.2f}",
                        "grad_norm_last": f"{grad_norm_last:.6f}",
                        "grad_norm_max": f"{grad_norm_max:.6f}",
                        "loss_scale": f"{scale_after_step:.1f}" if scaler.is_enabled() else "",
                        "skipped_steps": skipped_steps,
                    }
                )
                print(
                    f"step {step:5d} | train loss {train_loss:.4f} | "
                    f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | "
                    f"gpu {gpu_mem_mb:.1f}/{gpu_peak_mb:.1f} MB | "
                    f"grad {grad_norm_last:.3f}/{grad_norm_max:.3f} | "
                    f"updates {successful_steps}/{window_steps}"
                )

            if should_eval:
                val_loss = estimate_loss(
                    model,
                    val_batcher,
                    cfg.train.eval_batches,
                    runtime,
                    debug_checks=args.debug_checks,
                )
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                logger.log(
                    {
                        "step": step,
                        "optimizer_step": optimizer_step,
                        "split": "val",
                        "loss": f"{val_loss:.6f}",
                        "lr": f"{lr:.8f}",
                        "tokens_per_sec": "",
                        # 当前没有单独测量 validation 显存，留空比复用训练窗口数据更准确。
                        "gpu_mem_mb": "",
                        "gpu_peak_mb": "",
                        "grad_norm_last": "",
                        "grad_norm_max": "",
                        "loss_scale": f"{scale_after_step:.1f}" if scaler.is_enabled() else "",
                        "skipped_steps": "",
                    }
                )
                print(f"          | val loss   {val_loss:.4f} | best {best_val_loss:.4f}")

            if should_checkpoint:
                numbered_path = out_dir / "checkpoints" / f"step_{step:06d}.pt"
                latest_path = out_dir / "latest.pt"
                checkpoint_payload = build_training_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    optimizer_step=optimizer_step,
                    config_dict=config_dict,
                    best_val_loss=best_val_loss,
                    train_batcher=train_batcher,
                    val_batcher=val_batcher,
                    tokenizer_hash=tokenizer_hash,
                    data_hash=data_hash,
                    vocab_size=tokenizer.vocab_size,
                    tokenizer_path=tokenizer_path,
                )
                save_checkpoint(numbered_path, checkpoint_payload)
                latest_method = update_latest_checkpoint(latest_path, numbered_path)
                print(f"[checkpoint] saved {numbered_path}")
                print(f"[checkpoint] updated {latest_path} via {latest_method}")

            if should_close_window and step < cfg.train.max_steps:
                window_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                window_micro_batches = 0
                window_tokens = 0
                window_steps = 0
                window_optimizer_step_start = optimizer_step
                window_grad_norm_last = torch.zeros((), dtype=torch.float32, device=device)
                window_grad_norm_max = torch.zeros((), dtype=torch.float32, device=device)
                runtime.reset_peak_memory()
                interval_timer.start()

    if args.sample:
        engine = InferenceEngine(MiniGPTModelRunner(model, runtime), tokenizer)
        result = engine.generate(
            "MiniGPT",
            GenerationConfig(max_new_tokens=80, strategy="sample", temperature=0.9, seed=cfg.train.seed),
        )
        print("=" * 80)
        print("Sample:")
        print(result.full_text)


if __name__ == "__main__":
    main()
