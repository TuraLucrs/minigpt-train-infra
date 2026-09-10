# v0.9 后端、Profiler 与可恢复多卡矩阵

## 1. 交付状态与研究边界

本版本把 CPU/Gloo、CUDA/NCCL、Ascend/HCCL 的运行时与 Profiler 适配、真实 benchmark
入口、矩阵执行器、恢复和证据校验一起交付。它可以执行真实模型，不是仅生成实验计划。
软件验证使用 tiny Qwen3 和 CPU；正式性能只接受真实 Qwen3-32B dense 和实际 accelerator。
当前硬件验证仍待执行，不能据此声称 A3、A5 或 CUDA 上已获得任何吞吐提升。

v0.8 的软件检查点为 `2a661b1372b521d04c26fefc91cfc6080c72054b`，对应
[Linux CPU/Gloo CI](https://github.com/TuraLucrs/minigpt-train-infra/actions/runs/34517516717)
已通过。v0.9 基于该提交继续开发。按本轮明确授权，两版软件可以先于 v0.8 真机 A/B 完成；
这不会把 v0.8 候选 argmax 自动升级成默认路径，也不会改写 v0.7/v0.7.1 冻结证据。

## 2. 后端边界与真实指标

核心模型、KV Cache、调度器继续使用 PyTorch Tensor 和原有 runner 接口。设备差异集中于
`src/minigpt/backends/`，不包装普通数学算子。

| 层 | 入口 | 职责 |
|---|---|---|
| 运行时 | `RuntimeContext` / `backends.runtime` | 设备选择、精度、autocast、同步、RNG、event、allocator 内存和版本 |
| 分布式 | `DistributedContext` / `TensorParallelReplicaContext` | Gloo/NCCL/HCCL、TP 子组、collective 与 rank 身份 |
| Profiler | `backends.profiling` | 实际采集、导出、manifest、解析与指标能力 |
| 设备空闲检查 | `backends.telemetry` | 模型加载前的多轮 npu-smi / nvidia-smi 观测 |
| benchmark 契约 | `benchmark_contract` | 输入/输出摘要、真实设备映射、逐 rank 测量内存 |
| 实验执行 | `experiment_matrix` | 配置展开、子进程、故障恢复、原始产物校验与 A/B 汇总 |

`torch_npu` 按需在 Ascend 后端加载。CPU 和 CUDA 的正常导入/执行不依赖可用的 Ascend SDK。
显式要求某 accelerator 或精度时，不静默回退到 CPU/FP32。CPU 的可见 accelerator 数为 0，
它的进程编号不代表物理卡。

新报告使用 `memory_measurements`：单位为 **bytes**，来源为 PyTorch allocator，并区分
allocated、reserved、peak allocated、peak reserved 和 total capacity。每个 measured run
结束时先保存 `memory_snapshot`，再启动独立 profile replay；汇总通过 int64 collective
保留逐 rank 字节值。`sum_rank_peak_allocated_bytes` 是各 rank 各自峰值之和，不是同一时刻
整机内存的实测峰值。旧 `*_mb` 字段为 MiB 兼容接口；CPU 的旧 0 哨兵不作为测量值使用，
新口径中 CPU allocator、缺失的 reserved counter 等明确为 `null` / unsupported。

设备编号也必须可核验。矩阵在启动 Python 前设置实际设备可见性，入口将其与
`--logical-device-ids`、`LOCAL_RANK` 和 runtime device 对齐。当前矩阵是**单机多设备**；
不把 A3 TP16 描述成已经实现了多机启动。CUDA 数字映射按宿主 NVML/PCI 顺序处理，A3/A5
另记录物理卡与 chip 的显式对应关系。示例 device map 仅是填写样本，必须对照真实机器。

## 3. Profiler 两种格式与可比性

`infer_qwen3_continuous_batching.py --profile` 默认仍写旧 Ascend schema v1，供 v0.7.1/v0.8
原门禁使用。v0.9 显式加 `--profile-format portable`；TP 单请求/静态 batch 的新 profile
功能直接使用 portable schema v2。旧 `minigpt.ascend_profiling` 导入路径由兼容 shim 保留。

| 后端 | 实际采集 | 主要产物与限制 |
|---|---|---|
| CPU | `torch.profiler` CPU activity | Kineto trace、Host 阶段/算子、每 rank capture；设备计算与通信占比未知 |
| CUDA | `torch.profiler` CPU + CUDA activity | Kineto kernel/runtime/NCCL 事件、Host/device 关联与实际窗口 |
| Ascend | `torch_npu.profiler` | operator、kernel、step trace、timeline、communication 与每 rank capture |

v2 manifest 绑定全部选中 rank、backend、workload 的文件/语义哈希、代码提交、窗口协议和
各 artifact 的 SHA-256/大小。加载时重算这些值；缺文件、污染旧目录、不完整 rank 或窗口
均不能计为完整采集。matrix 还核对 profile replay 的原始请求输出、digest、measured 输出
和 profile/benchmark 身份。

所有 profile replay 在 measured repeats 之后单独运行，`measurement_excluded=true`。
报告保留 `aggregate_step_fractions` 和 `metric_capabilities`：未观测到或后端不支持的值
为 `null`，不能用 0 填补。特别是 CPU 没有 accelerator 指标；CUDA 事件不足以证明通信
不存在时，也不能把通信时间当作零。

Ascend step trace 的原生 Stage 与 Kineto 的 Host `ProfilerStep` 并非相同分母。跨后端可
比较具有相同口径的 TTFT、TPOT、完成吞吐和 allocator 内存；Profiler 百分比必须连同
time base、来源和能力解释。不能直接拿两个不同分母的占比宣称某硬件通信更少。

固定 batch 只有第 0 步包含 Prefill，因此矩阵使用 profiler `skip=0, warmup=0, active=4`
覆盖 Prefill 与 Decode。CPU 预设 active=3。普通 benchmark 已有独立 warmup，不能为了
Profiler warmup 跳过唯一的 Prefill 后又声称采到了 Prefill。

## 4. 严格 A/B 矩阵

| 预设 | logical devices | 物理卡关系 | 完整点数 |
|---|---|---|---:|
| `v09_ascend_a3.json` | 2 / 4 / 8 / 16 | 每卡 2 个 logical devices，即 1 / 2 / 4 / 8 卡 | 88 |
| `v09_ascend_a5.json` | 1 / 2 / 4 / 8 | 每卡 1 个 device | 88 |
| `v09_cuda.json` | 1 / 2 / 4 / 8 | 每个完整 GPU 对应一个 device | 88 |
| `v09_cpu_correctness.json` | 1 / 2 / 4 个进程 | 0 张 accelerator 卡 | 16 |

硬件预设各有 short 和 long-prefill 两类输入，每个 case 都是 A/B/B/A 四个独立进程，
每进程 1 次 warmup、3 次 measured、1 次独立 profile replay。CPU 使用每变体 1 个 session、
2 次 measured，仅用于验证执行链。可以分 case 运行，但未完成的点在全矩阵报告中仍为缺失。

| axis | A | B | 固定条件与用途 |
|---|---|---|---|
| KV | 单请求 recompute | 同请求 KV Cache | 固定模型、TP、prompt、输出长度；分别观察 Prefill/Decode |
| TP | 最小 TP（A3=2，其余=1） | 该预设更大的 TP | 同一请求、cached runner、相同输入输出工作量 |
| Batching | 静态 batch | 同 batch 的 continuous engine | 相同 TP、请求清单、同时到达、输出长度；比较批处理执行方式 |

Batching case 包括静态 cache 与 fixed-slot cache、调度 book-keeping 的实际差异；它测量
这两种执行方式，不能把结果全部归因于单独一个 scheduler 函数，也不代表生产线上随机
到达流量的吞吐。两种入口的阶段边界不同，因此该 case 的 Prefill/Decode 比值会根据
`phase_timing_scope` 判定是否可比，而不是强行相除。

所有点使用同一份 `WorkloadTrace` 的原始文件与语义哈希，greedy、固定 `max_new_tokens`、
不提前 EOS、arrival=0。TP 入口直接使用文件里的 GenerationConfig，不把空 EOS 设置替换
成 tokenizer 默认值。单请求和静态 batch 逐步同步测量 token 可用边界；静态 batch 中
每行独立记录 TTFT、TPOT 和结束时刻，提前 EOS 的行不会被余下行的 E2E 时间替代。

矩阵检查真实执行模式、协议、设备映射、模型与权重清单、软件环境、代码状态、完整请求
集合、逐轮输出、计算长度、原始 profile、ABBA 实际时序和跨 session 吞吐 CV。硬件默认
CV 上限 10%。CPU 时延只验证有限、定义明确和可以贯通执行，不用于硬件稳定性结论。
固定 batch 的 greedy 输出不一致会阻断比较；先查数值和 TP 归约顺序，再考虑性能。

Qwen3 TP 计划支持 KV heads 复制，因此 TP16 不是仅因 KV heads 少于 TP 就静态拒绝；
启动前仍用真实 config 校验维度和分片约束。内存不足、SDK/collective 不支持等由真实运行
保留失败记录，不会用理论模型容量或 dry-run 宣布成功。

## 5. 实际运行

先使用匹配硬件的 PyTorch、CUDA/NCCL 或 torch_npu/CANN 环境，然后在目标环境中安装项目：

```bash
python -m pip install --no-deps -e .
python benchmarks/run_v09_matrix.py --config configs/v09_ascend_a3.json --list-points
```

Ascend A3 完整执行示例：

```bash
export CANN_VERSION='填写目标机器确认的完整 CANN 版本'
python benchmarks/run_v09_matrix.py \
  --config configs/v09_ascend_a3.json \
  --model-dir /path/to/Qwen3-32B \
  --device-map /path/to/verified-a3-device-map.json \
  --interconnect-topology '填写实际物理卡、芯片与互连拓扑' \
  --output-dir runs/v09_a3
```

A5 使用 `configs/v09_ascend_a5.json` 及 A5 实际映射；CUDA 使用 `configs/v09_cuda.json` 及
GPU 实际映射。CUDA preflight 查询 `nvidia-smi`，NPU preflight 查询 `npu-smi`。至少三轮、
跨度至少 1 秒，检查整个窗口的设备内存占用 ≤10%、计算利用率 ≤5%。任何缺设备、缺值、
命令失败或超时均不能视为空闲，原始采集留在 attempt 目录供审计。

设备窗口有限时，先完成一个 case 的整个 ABBA，随后恢复：

```bash
python benchmarks/run_v09_matrix.py \
  --config configs/v09_ascend_a3.json \
  --model-dir /path/to/Qwen3-32B \
  --device-map /path/to/verified-a3-device-map.json \
  --interconnect-topology '与首次执行完全相同的拓扑说明' \
  --output-dir runs/v09_a3 \
  --case kv-short-tp2

# 相同命令加 --resume，并去掉 --case，继续其余点。
```

`--point` 可选择单点用于诊断，但该 case 的前序点必须已成功；缺前序会记为 deferred。
某点失败时，同 case 的后续点也延后，恢复后继续保持真实 AB/ABBA 顺序。正式 case 仍必须
具备完整 ABBA。`--dry-run` 只保存 pending 计划，不采集、不改变点的执行状态。
`--resume` 要求相同配置、代码内容、模型、设备映射和
执行环境；已成功产物先校验再复用。新代码或新协议使用新的输出目录，避免混成同一矩阵。

CPU 实际执行链：

```bash
python scripts/create_tiny_qwen3_fixture.py --output runs/v09_tiny_model
python benchmarks/run_v09_matrix.py \
  --config configs/v09_cpu_correctness.json \
  --model-dir runs/v09_tiny_model \
  --output-dir runs/v09_cpu
```

TP1 使用独立 Python 子进程，TP≥2 使用 `torchrun`。本机 Windows PyTorch 2.10 CPU wheel 的
Gloo TCP/UV transport 无法初始化，连 TP1 的 `torchrun` 也遇到 TCPStore/libuv 限制；在该环境只执行
`--case kv-short-tp1 --case batching-short-tp1` 可验证真实 CPU 子进程，完整 CPU TP2/TP4 在
Linux CI 验证。不要把这种局部成功报告为完整 CPU 矩阵通过。

汇总一个或多个后端：

```bash
python benchmarks/summarize_v09_matrix.py \
  --input runs/v09_a3 \
  --input runs/v09_a5 \
  --input runs/v09_cuda \
  --output runs/v09_comparison.json \
  --markdown runs/v09_comparison.md \
  --require-complete --require-formal
```

`--require-complete` 要求所有点和 case 的重新验证通过；`--require-formal` 另外要求干净源码、
真实 Qwen3-32B 和 accelerator 性能证据。CPU 正确性矩阵即使完整通过，也不满足后者。

## 6. 产物、恢复和软件验证

每个矩阵保留 `matrix_state.json`、固定 workloads 和逐 point/attempt 子目录。每个 attempt
保存真实命令、退出状态、起止时间、`benchmark.log`、输入/输出/Profiler 的哈希与摘要。
硬件 attempt 另保留 `device_preflight.json`，执行环境保留 `runtime_probe.json`。失败和中断
保留为历史 attempt；恢复不能覆盖此前证据。
汇总重新读取 hash-bound 原始产物，不能只信 journal 中写着 succeeded 或保存的指标。

Linux 每个子进程命令由持有矩阵锁的 subreaper supervisor 管理；它跟踪 launcher 及独立
session 中的后代，用 PID/进程创建标记核验身份。控制器被 SIGTERM 或 SIGKILL 结束后，
supervisor 继续清理并收养孤儿 worker；只有确认全部结束才释放锁。若无法完成回收，
状态为 `cleanup_blocked`，整个矩阵停止启动后续点。supervisor 本身异常退出且没有清理完成
记录时，持久登记也会阻断 resume。现场应检查 `.process_guards/` 和 attempt 内的
`*.process_tree.json`，确认原进程状态；删除锁或登记文件不构成清理完成的证据。
Windows 保留进程树终止逻辑和创建后的异常保护，Linux 的 subreaper/控制器 SIGKILL 保护
不属于 Windows 已验证能力。

运行结束后保留 `matrix_summary.json` 与 `MATRIX_REPORT.md`。代码没有把 tiny、CPU 或缺点
矩阵升级为真实性能结论；`complete`、`formal_performance_evidence` 和每个可比指标应一起
阅读。跨后端比值要求模型、workload、计算工作量、源码内容、精度、warmup/repeats、SLO、
profile/capacity 合同和指标语义一致，并保留后端环境差异。

新增回归入口：

```bash
python tests/test_backends.py
python tests/test_profiler_backends.py
python tests/test_benchmark_portable.py
python tests/test_backend_telemetry.py
python tests/test_benchmark_entrypoints.py
python tests/test_experiment_matrix.py
```

Linux CI 设置 `MINIGPT_RUN_GLOO_TESTS=1`，同时运行全部历史训练/推理/服务/v0.8 回归和这些
新增测试。CPU Profiler、CLI 和矩阵测试都执行真实 tiny 推理；GPU/NPU SDK 契约与管理命令
解析测试不能替代实际硬件采集。最终提交、CI 链接、恢复材料和待跑项见
[v0.8 / v0.9 软件交接](MINIGPT_HANDOFF_2026-09-11_V08_V09_SOFTWARE_READY.md)。
