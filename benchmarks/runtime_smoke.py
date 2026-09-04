"""验证 CPU/CUDA/Ascend Runtime 的设备、autocast、同步、Event 和内存接口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.runtime import DeviceIntervalTimer, RuntimeContext  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 Runtime 后端 smoke test 并输出 JSON。"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "npu"), default="auto"
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--output", default="runs/runtime_smoke.json")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    if args.matrix_size <= 0:
        raise ValueError("matrix-size 必须大于 0")
    runtime = RuntimeContext.create(
        args.device,
        args.precision,
        allow_accelerator_fallback=False,
        allow_precision_fallback=False,
    )
    runtime.manual_seed(2026)
    runtime.empty_cache()
    runtime.reset_peak_memory()
    left = torch.randn(args.matrix_size, args.matrix_size, device=runtime.device)
    right = torch.randn(args.matrix_size, args.matrix_size, device=runtime.device)
    timer = DeviceIntervalTimer(runtime)
    timer.start()
    with runtime.autocast():
        result = left @ right
    elapsed = timer.elapsed_seconds()
    runtime.synchronize()
    current_mb, peak_mb = runtime.memory_stats_mb()
    report = {
        "status": "passed",
        "backend": runtime.backend_metadata(),
        "matrix_size": args.matrix_size,
        "result_dtype": str(result.dtype),
        "result_isfinite": bool(torch.isfinite(result).all().item()),
        "wall_seconds": elapsed,
        "device_seconds": timer.last_device_seconds,
        "current_memory_mb": current_mb,
        "peak_memory_mb": peak_mb,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告：{output}")


if __name__ == "__main__":
    main()
