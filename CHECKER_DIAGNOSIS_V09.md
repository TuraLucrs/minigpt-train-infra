# v0.9 紧凑矩阵 checker 诊断（Atlas A3，2026-09-11）

## 1. 现象

`run_v09_matrix.py --config configs/v09_ascend_a3.json`（commit 4a85f42，8 逻辑 NPU / 4 卡双芯）
首轮 12 点执行中，每个 case 的 a0 session 在 **benchmark 全部成功结束后**被矩阵层判
`failed`，同 case 其余 session 被 defer。典型日志：benchmark 进程 exit 0、输出
report.json（TTFT/TPOT/吞吐/内存齐全），但矩阵状态记
`reason = "ValueError: benchmark software environment differs from the saved interpreter identity"`。

## 2. 根因

`src/minigpt/experiment_matrix.py` 的软件身份比对（4a85f42 上位于 1119-1121 行附近）：

- 驱动侧 `_software_identity()` 经 `_SOFTWARE_CODE` 用 **importlib.metadata** 记录 torch 版本
  → 本机得到 `"2.10.0"`；
- 报告侧 `report["environment"]["torch"]` 来自运行时 **torch.__version__**
  → 本机为 `"2.10.0+cpu"`。

本容器 torch 为源码构建：dist metadata Version=2.10.0，`torch.__version__=2.10.0+cpu`。
两个不同来源对同一安装给出不同字符串，逐字符串相等比较必然失败。**任何带 local version
段（+cpu/+git 等）的 torch 环境上，所有 accelerator session 都会在 benchmark 成功后被误判**；
官方 CPU CI 用标准 wheel（两来源一致），因此 CI 无法暴露此路径。

## 3. 逐字段核对（kv-long-tp2-a0，attempt-000）

| 比对项 | 驱动 identity | 报告 | 结果 |
|---|---|---|---|
| python | 3.12.13 | 3.12.13 | 一致 |
| torch | 2.10.0（metadata） | 2.10.0+cpu（__version__） | **唯一不一致（同一安装）** |
| torch-npu | 2.10.0.post5.dev20260821 | 2.10.0.post5.dev20260821 | 一致 |
| git commit | 4a85f42 | 4a85f42 | 一致 |
| device_mapping | verified=True | verified=True, ids [0,1] | 一致 |
| rank logical/global IDs | [0,1] / [0,1] | [0,1] / [0,1] | 一致 |
| physical_card_count / chips_per_card | 1 / 2 | 1 / 2 | 一致 |
| interconnect_topology / run_label | 与计划一致 | 与计划一致 | 一致 |
| protocol（warmup/repeats/decode） | 1/3/recompute | 1/1/3/recompute | 一致 |
| 模型 provenance（config/weights 哈希） | 一致 | 一致 | 一致 |

kv-long-tp2-a0 实测（3 次 measured 中位）：TPOT 149.364 ms，TTFT 150.800 ms，
E2E 2392.028 ms，model load 18.503 s，max rank peak 31516.154 MB，exit 0。
数据本身有效，仅被该检查误拒。

## 4. 修复

分支 `fix/v09-software-identity-local-version`，commit `736dd7b`（基于 4a85f42）：
新增 `_public_version()`（剥离 `+local` 段），比对改为
`_public_version(report torch) != _public_version(identity torch)`；
None 安全。python、torch_npu 及其余校验不变。复跑输出目录：
`runs/v09_a3_compact_736dd7b`（目录名与实际运行代码一致）。

## 5. 首轮现场保留

`runs/v09_a3_compact_4a85f42/`（首轮 3 个 a0 session 完整产物 + matrix_state.json）
作为本 bug 的现场证据保留并随包推送；该目录不是正式验收结果。
