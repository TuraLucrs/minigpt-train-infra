# v0.6 Qwen3-32B Tensor Parallel 实机验收 — 运行记录（RUN LOG）

- 机器：Ascend Atlas A3（8 物理卡 × 2 芯 = 16 逻辑 Ascend910，64GB HBM/芯），驱动 26.2.rc1.b021，
  宿主机 CANN 9.1，容器 `py312_anytest` CANN 9.2
- 执行位置：容器内 `/root/minigpt-train`（bundle `upgrade/v0.6-qwen3-tensor-parallel` 克隆）
- 代码 commit：`ea672a6f31c06fb5cad6ee474203a695b6d2bab3`（基线 `bc4e8b5` RC2 + 修复 `ea672a6`），树干净
- 环境：Python 3.12.13 / torch 2.10.0+cpu / torch_npu 2.10.0.post5.dev20260821 / transformers 5.14.1，backend HCCL
- 权重：ModelScope `Qwen/Qwen3-32B`，17 分片 BF16 ≈62GB，全部 SHA-256 记录于各报告 `provenance.weights`
- 时间：2026-09-04，本地时间 UTC+8（报告内 `created_at_utc` 为 UTC）

## 每次运行前的固定环境（容器内）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 不 source 则 torch_npu 找不到 libhccl.so
export HCCL_NPU_SOCKET_PORT_RANGE=60000-60050        # 默认端口 16666 与其他任务冲突（EI0020）
cd /root/minigpt-train
# 容器 hostname node-21-123 不可解析，已在容器 /etc/hosts 追加 '127.0.0.1 node-21-123'
# torchrun 一律用静态 rendezvous（--standalone 的动态 TCPStore 依赖 hostname 解析，会 300s 超时）
```

## 阶段 1：CPU 回归 + Gloo 门禁（12:05，console_logs/tests.log）

```bash
# v0.1–v0.5 全部回归套件 + v0.6 线程仿真
python tests/test_core.py && python tests/test_inference.py && python tests/test_kv_cache.py \
  && python tests/test_qwen3.py && python tests/test_qwen3_tp.py && python tests/test_reference_parity.py \
  && python tests/test_resume_consistency.py && python tests/test_tp_scaling.py
# Gloo 真实进程组门禁（此前在开发容器被平台权限阻塞）
MINIGPT_RUN_GLOO_TESTS=1 python tests/test_qwen3_tp.py
```
结果：全部通过（"v0.6 Gloo process-group tests passed"）。

## 阶段 2：NPU 单芯 Runtime smoke（12:05）

```bash
python benchmarks/runtime_smoke.py --device npu --precision bf16
# → runs/runtime_smoke_npu.json（显式 npu/bf16，无回退）
```

## 阶段 3：tiny HCCL hardware smoke + collectives，TP=2/4/8（12:16–12:19）

每档先 smoke 后 collective，端口每档换新（29512–29514 区间内取空闲）：

```bash
torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=<port> \
  benchmarks/tp_hardware_smoke.py --device npu --backend hccl --precision bf16 \
  --output runs/tp2_hardware_smoke.json
torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=<port> \
  benchmarks/benchmark_tp_collectives.py --device npu --backend hccl --precision bf16 \
  --message-mb 0.25 --message-mb 1 --message-mb 4 \
  --physical-card-count 1 --chips-per-card 2 \
  --interconnect-topology "Atlas A3 card0, 2 chips, HCCS intra-card" \
  --output runs/tp2_collectives.json
# TP=4: --nproc-per-node=4, physical-card-count 2, topology "Atlas A3 cards0-1, 2 chips/card, HCCS intra-card + cross-card board interconnect"
# TP=8: --nproc-per-node=8, physical-card-count 4, topology "Atlas A3 cards0-3, 2 chips/card, HCCS intra-card + cross-card board interconnect"
```
结果：6 份报告全部通过（smoke=correctness，collective=formal_collective_measurement）。

## 阶段 4：32B 短生成（14:11，console_logs/tp2_32b_short.log）

```bash
torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29512 \
  infer_qwen3_tp.py --model-dir /data/models/Qwen3-32B --device npu --backend hccl --precision bf16 \
  --chat-template --max-new-tokens 8
```
首次运行暴露 transformers 5.x `apply_chat_template` 返回 `BatchEncoding` 的兼容问题，
修复为 commit `ea672a6`（encode() 归一化为 list[int]），回归重跑通过。修复后输出：
"你好！我是通义千问，…"。

## 阶段 5：32B 正式基准 TP=2/4/8（14:12–14:23，同一 prompt/参数/commit/权重）

```bash
# TP=2（console_logs/bench_tp2.log）
torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29512 \
  benchmarks/infer_qwen3_tp.py \
  --model-dir /data/models/Qwen3-32B --device npu --backend hccl --precision bf16 \
  --prompt "你好，请介绍一下你自己。" --chat-template --max-new-tokens 32 --warmup 2 --repeats 5 \
  --hash-weights --physical-card-count 1 --chips-per-card 2 \
  --interconnect-topology "Atlas A3 card0, 2 chips, HCCS intra-card" \
  --run-label qwen3-32b-tp2 --output runs/qwen3_32b_tp2.json

# TP=4（console_logs/bench_tp4.log）——同上，仅改：
#   --nproc-per-node=4 --master-port=29513 --physical-card-count 2
#   --interconnect-topology "Atlas A3 cards0-1, 2 chips/card, HCCS intra-card + cross-card board interconnect"
#   --run-label qwen3-32b-tp4 --output runs/qwen3_32b_tp4.json

# TP=8（console_logs/bench_tp8.log）——同上，仅改：
#   --nproc-per-node=8 --master-port=29514 --physical-card-count 4
#   --interconnect-topology "Atlas A3 cards0-3, 2 chips/card, HCCS intra-card + cross-card board interconnect"
#   --run-label qwen3-32b-tp8 --output runs/qwen3_32b_tp8.json
```
三份报告 evidence_class 均为 `formal_qwen3_32b_tp_hashed`；生成文本逐字一致；
rank 结果一致；参数占比精确 1/P；TTFT/TPOT/吞吐/HBM 见下表。

## 阶段 6：Scaling 汇总（14:28）

```bash
python benchmarks/summarize_tp_scaling.py \
  --report runs/qwen3_32b_tp2.json --report runs/qwen3_32b_tp4.json --report runs/qwen3_32b_tp8.json \
  --output runs/qwen3_32b_tp_scaling.json
```
结果：`formal_qwen3_32b_tp_scaling`，baseline_world_size=2，same_git_commit=True，same_weight_manifest=True。

## 附加：复验 smoke（14:46–14:50，应用户要求现场演示 NPU 在跑）

```bash
torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29515/29516/29517 \
  benchmarks/tp_hardware_smoke.py --device npu --backend hccl --precision bf16 \
  --output /tmp/tp2_reproof_smoke.json
```
3 次全部 exit 0；采样观测 chip0 HBM 3142→3430MB（rank0 绑定设备瞬间）。
留档：`reproof_tp2_hardware_smoke_1450.json`。

## 结果摘要（中位数）

| TP | TTFT ms | TPOT ms | E2E ms | 输出 tok/s | 单 rank 峰值 HBM | 扩展效率 |
|----|---------|---------|--------|-----------|------------------|----------|
| 2（基线） | 162.1 | 172.1 | 5494 | 5.824 | 30.7 GB | 1.000 |
| 4 | 188.9 | 174.2 | 5602 | 5.712 | 15.3 GB | 0.490 |
| 8 | 198.0 | 178.1 | 5713 | 5.601 | 7.9 GB | 0.240 |

单请求 decode 为通信主导，吞吐随 TP 近似持平、TTFT 略升；TP=4/8 的收益在显存（3.9× 峰值释放），
有效利用率提升属 v0.7 真实并发范围（与 v0.6 文档 §9 边界一致）。

## 产物索引

| 文件 | 内容 |
|---|---|
| `qwen3_32b_tp{2,4,8}.json` | 正式基准报告（内嵌完整命令 argv、权重 SHA-256、commit、拓扑、逐 rank HBM、5 次重复逐次数据） |
| `qwen3_32b_tp_scaling.json` | 汇总报告 |
| `tp{2,4,8}_hardware_smoke.json` / `tp{2,4,8}_collectives.json` | tiny smoke / collective 原始数据 |
| `runtime_smoke_npu.json` | 单芯 runtime smoke |
| `reproof_tp2_hardware_smoke_1450.json` | 复验 smoke |
| `console_logs/` | 11 份控制台日志（tests/smoke/collectives/32B 短生成/三档基准） |
| `minigpt-train-v06-branch.bundle` | 完整分支历史（含修复 commit），`git bundle verify` 通过；恢复：`git clone <bundle> -b upgrade/v0.6-qwen3-tensor-parallel` |
| `RUN_LOG.md` | 本文档 |

## 待办（项目侧）

- commit `ea672a6` 推送 GitHub 并核验远端 SHA（需项目所有者凭据；可用上述 bundle）；
- 按文档 §8 复核后决定最终 v0.6 tag。
