"""TP Scaling 汇总公式、可比性门禁和证据分级测试。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from summarize_tp_scaling import summarize_scaling  # noqa: E402


def fake_report(
    world_size: int,
    *,
    ttft_ms: float,
    tpot_ms: float,
    e2e_ms: float,
    throughput: float,
    peak_mb: float,
) -> dict[str, object]:
    return {
        "benchmark": "single_request_npu_qwen3_tp_kv_cache",
        "summary": {
            "ttft_ms": {"median": ttft_ms},
            "tpot_ms": {"median": tpot_ms},
            "e2e_latency_ms": {"median": e2e_ms},
            "output_tokens_per_second": {"median": throughput},
        },
        "distributed": {
            "backend": "hccl",
            "chips_per_card": 2,
            "world_size": world_size,
            "max_rank_median_request_peak_mb": peak_mb,
            "sum_rank_median_request_peak_mb": peak_mb * world_size,
        },
        "environment": {
            "device_type": "npu",
            "device_name": "Ascend 950",
            "precision": "bf16",
            "torch": "2.10.0",
            "torch_npu": "2.10.0.post2",
            "total_memory_mb": 65536.0,
        },
        "provenance": {
            "config_sha256": "config",
            "weights": [{"name": "model.safetensors", "sha256": "weights"}],
            "git": {"commit": "abc123", "dirty": False},
        },
        "request": {
            "prompt": "hello",
            "max_new_tokens": 32,
            "warmup": 2,
            "repeats": 5,
        },
        "model": {"config": {"hidden_size": 5120}},
        "workload_fingerprint": {
            "prompt_sha256": ["prompt"],
            "encoded_prompt_lengths": [1],
            "system_prompt_sha256": None,
            "use_chat_template": False,
            "enable_thinking": False,
            "decode_mode": "kv_cache",
        },
        "evidence_class": "formal_qwen3_32b_tp_hashed",
    }


def main() -> None:
    tp2 = fake_report(
        2,
        ttft_ms=100.0,
        tpot_ms=10.0,
        e2e_ms=200.0,
        throughput=10.0,
        peak_mb=40_000.0,
    )
    tp4 = fake_report(
        4,
        ttft_ms=60.0,
        tpot_ms=6.0,
        e2e_ms=120.0,
        throughput=18.0,
        peak_mb=22_000.0,
    )
    summary = summarize_scaling([(Path("tp4.json"), tp4), (Path("tp2.json"), tp2)])
    assert summary["world_sizes"] == [2, 4]
    assert summary["baseline_world_size"] == 2
    assert summary["evidence_class"] == "formal_qwen3_32b_tp_scaling"
    row = summary["rows"][1]
    assert abs(row["latency"]["ttft_ms"]["speedup_vs_baseline"] - 5 / 3) < 1e-12
    assert abs(row["latency"]["ttft_ms"]["scaling_efficiency"] - 5 / 6) < 1e-12
    assert abs(row["throughput"]["speedup_vs_baseline"] - 1.8) < 1e-12
    assert abs(row["throughput"]["scaling_efficiency"] - 0.9) < 1e-12
    assert abs(row["memory"]["per_rank_reduction_vs_baseline"] - 20 / 11) < 1e-12

    incompatible = fake_report(
        8,
        ttft_ms=50.0,
        tpot_ms=5.0,
        e2e_ms=100.0,
        throughput=25.0,
        peak_mb=15_000.0,
    )
    incompatible["request"]["max_new_tokens"] = 64
    try:
        summarize_scaling([(Path("tp2.json"), tp2), (Path("tp8.json"), incompatible)])
    except ValueError:
        pass
    else:
        raise AssertionError("请求配置不同的报告必须拒绝比较")

    incompatible_environment = fake_report(
        8,
        ttft_ms=50.0,
        tpot_ms=5.0,
        e2e_ms=100.0,
        throughput=25.0,
        peak_mb=15_000.0,
    )
    incompatible_environment["environment"]["precision"] = "fp16"
    try:
        summarize_scaling(
            [(Path("tp2.json"), tp2), (Path("tp8.json"), incompatible_environment)]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("设备或精度不同的报告必须拒绝比较")

    incompatible_protocol = fake_report(
        8,
        ttft_ms=50.0,
        tpot_ms=5.0,
        e2e_ms=100.0,
        throughput=25.0,
        peak_mb=15_000.0,
    )
    incompatible_protocol["request"]["repeats"] = 3
    try:
        summarize_scaling(
            [(Path("tp2.json"), tp2), (Path("tp8.json"), incompatible_protocol)]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("warmup/repeats 不同的报告必须拒绝比较")

    print("v0.6 TP scaling summary tests passed.")


if __name__ == "__main__":
    main()
