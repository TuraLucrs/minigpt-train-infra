# RUN NOTE — v0.9 compact matrix on Atlas A3 (runs/v09_a3_compact_736dd7b)

## 1. Environment

- Host: node-21-123, Atlas A3, 8 physical dual-chip cards; run used 4 cards -> 8 logical NPUs
  (device map `verified-a3-device-map.json`, logical_device_id = physical_card_id * 2 + chip_id,
  cross-checked against `npu-smi info -m`; `chips_per_card=2`).
- Container: py312_anytest; CANN toolkit 26.2.rc1.b021 (`ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0`,
  `set_env.sh` sourced before every python invocation).
- Stack (from the recorded software identity): Python 3.12.13, torch 2.10.0 (importlib.metadata) /
  2.10.0+cpu (`torch.__version__`), torch-npu 2.10.0.post5.dev20260821, transformers 5.14.1,
  tokenizers 0.22.2, safetensors 0.8.0, numpy 2.2.6.
- Model: /data/models/Qwen3-32B (config sha256 97e295b6... recorded in matrix_state identity).
- Code: branch `upgrade/v0.9-backends-profiler-matrix` @ 4a85f42 first probed; fixed tree on
  `fix/v09-software-identity-local-version` (736dd7b fix + 51d8740 docs; run identity records 51d8740, dirty only for
  the untracked device-map file).
- Config: `configs/v09_ascend_a3.json` — 12 independent sessions, 3 ABBA cases,
  sessions_per_variant=2, warmup=1, repeats=3, max_seq_len 4096, profile skip=0/warmup=0/active=4, seed 2026.

## 2. Commands

First round (bug probe, output-dir runs/v09_a3_compact_4a85f42):

```bash
export CANN_VERSION=26.2.rc1.b021
python benchmarks/run_v09_matrix.py --config configs/v09_ascend_a3.json \
  --model-dir /data/models/Qwen3-32B \
  --device-map /root/minigpt-v09/verified-a3-device-map.json \
  --interconnect-topology "4 physical cards, 2 logical devices per card" \
  --output-dir runs/v09_a3_compact_4a85f42
```

Fixed rerun (this directory; per-session driver command is recorded in every attempt entry, e.g.
`torchrun --nproc-per-node=8 ... benchmarks/infer_qwen3_tp.py --model-dir /data/models/Qwen3-32B ...`):

```bash
export CANN_VERSION=26.2.rc1.b021
python benchmarks/run_v09_matrix.py --config configs/v09_ascend_a3.json \
  --model-dir /data/models/Qwen3-32B \
  --device-map /root/minigpt-v09/verified-a3-device-map.json \
  --interconnect-topology "4 physical cards, 2 logical devices per card" \
  --output-dir runs/v09_a3_compact_736dd7b
```

Driver console logs are included as DRIVER_CONSOLE.log in each run directory.

## 3. Result (12/12 succeeded, 0 OOM, 0 preserved failures)

| A/B case | Complete | Baseline token/s | Candidate token/s |
|---|---|---:|---:|
| kv-long-tp2 (long prefill: recompute vs KV cache, TP2) | True | 6.73849 | 5.36011 |
| tp-short-2-vs-8 (short decode: TP2 vs TP8) | True | 5.80095 | 5.79604 |
| batching-short-tp8 (short decode: static batch vs continuous engine) | True | 21.2728 | 20.9877 |

Evidence class: incomplete_or_development_matrix. Successful points: 12/12.

## 4. First-round checker failure and the fix (why this branch exists)

- First round @ 4a85f42: every benchmark succeeded but the matrix driver marked all accelerator
  sessions failed and deferred their case siblings. Evidence preserved in `runs/v09_a3_compact_4a85f42`.
- Root cause: `experiment_matrix.py` compared the report environment (which records `torch.__version__`,
  here "2.10.0+cpu") against the software identity (importlib.metadata, here "2.10.0") with strict string
  equality; the local build segment disagreed and every session failed the identity check. CPU CI cannot
  catch this because a standard wheel install makes both sources agree.
- Fix: commit 736dd7b wraps both sides with `_public_version()` (strip everything from "+").
- Diagnosis detail: `CHECKER_DIAGNOSIS_V09.md`.
- Regression: `python tests/test_experiment_matrix.py --unit-only` and `python tests/test_benchmark_entrypoints.py`
  both pass on the fixed tree.

## 5. Packaging layers (what is and is not in git)

Included (verification chain + reports):

- MATRIX_REPORT.md, matrix_summary.json, matrix_state.json, matrix_config.json, matrix_plan.json, DRIVER_CONSOLE.log
- workloads/ (generated workload JSONs and hashes)
- per attempt: benchmark.log, preflight.log, device_preflight.json, runtime_probe.json, *process_tree.json,
  output/{report.json, profile_manifest.json, source_workload.json}
- profiler derived summaries: ASCEND_PROFILER_OUTPUT CSVs (per-op/kernel summaries) and points/<id>/profiler/ summaries

Excluded (too large for git hosting; regenerable from the same workload/seed):

- trace_view.json — 60 files, ~7.5GB, all >100MB (GitHub hard limit)
- timeline/kernel DBs *.db — 120 files, ~2.2GB
- raw PROF_* device trees (~2.0GB) and FRAMEWORK raw op dumps
- bug-evidence dir additionally excludes ASCEND_PROFILER_OUTPUT CSVs (2.0GB); its crash evidence is the
  logs/reports/state files, which are included

Integrity: `SHA256SUMS` (repo root) covers every committed run artifact. Verify after clone:
`sha256sum -c SHA256SUMS` (paths are repo-root relative).
