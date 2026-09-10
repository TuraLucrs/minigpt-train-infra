# MiniGPT 推理 Infra 交接：v0.8 / v0.9 软件就绪

> 日期：2026-09-11（Asia/Shanghai）  
> 仓库：https://github.com/TuraLucrs/minigpt-train-infra  
> 本文记录软件交付与恢复入口；真实硬件验收、性能结论和正式版本 tag 仍待完成。

## 0. 新会话先读

1. 用户已经明确授权：从 `39448e35a6a5a9b8bb146e46d058692534500f09` 继续完成 v0.8、v0.9 的软件、测试、交接和提交推送，做到只剩真机运行与验收。无需再次询问是否开始开发或推送。
2. v0.8 软件已提交至 `upgrade/v0.8-decode-critical-path`，提交为 `2a661b1372b521d04c26fefc91cfc6080c72054b`；真实 Linux CPU/Gloo CI 已通过。
3. v0.9 软件已提交至 `upgrade/v0.9-backends-profiler-matrix`，软件提交为 `5e9efbba4473f54bb6e5c31beead4f4bc263179e`；真实 Linux CPU/Gloo、Profiler、推理 CLI 和完整 16 点 CPU 矩阵 CI 已通过。本文随后的交接提交只补充文档，最终交付 SHA、CI 和文件哈希见同批交付的 `DELIVERY_RECEIPT.json`。
4. 两版都没有真实 A3/A5/CUDA 新实验结果；不能把软件 CI、tiny 模型、dry-run 或管理命令 fixture 写成真机验收或性能提升。没有创建 v0.8/v0.9 正式硬件验收 tag。
5. v0.8 默认仍为 `full_gather`。候选 `distributed_argmax` 已具备正确性和实验门禁，是否有效必须由实际 ABBA 数据判定。
6. v0.7/v0.7.1 原 tag 未改写。旧 mixed `4×TP2 / TP8 = 4.24×` 受并发设备负载污染；后续干净 ABBA+BAAB 复现为约 `1.334×`，不能再使用旧夸大倍率。
7. 当前代码在 Windows 工作区的 `work/minigpt-train-infra`。它是 sparse、blob-filtered 仓库；历史大型 evidence 未全部物化，不能称为完整离线 Git 备份。恢复材料的覆盖范围见第 8 节。

## 1. 权威 refs 与软件验证

| 用途 | 分支 / tag | 提交 |
|---|---|---|
| 本轮起点 | v0.8 WIP | `39448e35a6a5a9b8bb146e46d058692534500f09` |
| v0.8 软件检查点 | `upgrade/v0.8-decode-critical-path` | `2a661b1372b521d04c26fefc91cfc6080c72054b` |
| v0.9 软件检查点 | `upgrade/v0.9-backends-profiler-matrix` | `5e9efbba4473f54bb6e5c31beead4f4bc263179e` |
| 已冻结 v0.7 | `v0.7-continuous-batching` 解引用 | `e20cc34097a68dd4f7791d92d24ec9d095fc7f57` |
| 已冻结 v0.7.1 | `v0.7.1-ascend-profiling-gate` 解引用 | `f7956b2a1c429587b74f4c4963ea7552f2c39dbd` |
| mixed 纠偏 | `investigate/v0.7.1-mixed-repro` | `beeb110eb781a8cb9c535662b33057eebf76de87` |

提交作者和提交者均为 `TuraLucrs <Lucifer24kl@gmail.com>`。GitHub 凭据已通过本机 Credential Manager 配置，remote URL 不含凭据。新机器使用自己的正常 GitHub 认证方式；不要把任何凭据复制进交接、源码或命令日志。

验证记录：

- v0.8 精确 SHA CI：[34517516717](https://github.com/TuraLucrs/minigpt-train-infra/actions/runs/34517516717)，静态检查及完整 Linux CPU/Gloo 回归成功。
- v0.9 软件 SHA CI：[34528747966](https://github.com/TuraLucrs/minigpt-train-infra/actions/runs/34528747966)，历史 14 个测试脚本和新增 6 个脚本成功；完整 CPU 矩阵为 8 cases、16 个真实进程点，含 TP2/TP4 Gloo。
- Windows 开发期 17 项历史/后端回归全部通过；另有真实 CLI、设备遥测解析和矩阵集成通过。交付证据保留命令、退出码及可用日志。
- Windows 本机 PyTorch 2.10 CPU wheel 无法初始化 Gloo TCP/UV transport，TP1 的 torchrun 也遇到 TCPStore/libuv 限制。TP1 矩阵使用独立 Python 子进程；真实 TP2/TP4 由 Linux CI 验证。没有把本地 Gloo 失败改记为通过。

软件测试环境：Python 3.12、torch 2.10.0+cpu、numpy 2.5.3、safetensors 0.8.0、transformers 5.16.1，项目版本 `0.9.0`。真机需要与目标驱动匹配的 CUDA/NCCL 或 torch_npu/CANN/HCCL 环境。

## 2. v0.8 完成内容

核心仍是单变量 Decode 通信 A/B：完整词表 AllGather 对照 distributed greedy argmax。

- rank-local LM Head 后只交换每行两个 FP32 值 `(score, global_token_id)`，保持最低 token id 的 tie-break；支持 FP16/BF16/FP32，拒绝会改变语义的 FP64 与不可精确表示的过大 token id。
- engine 首次运行前，各 rank 对路径与能力达成一致；runner 配置只读，避免热路径动态改设置造成 collective 次序不一致。
- sampling batch 自动回退 full logits，覆盖真实非退化采样、独立请求 RNG、整段生成序列、NaN/Inf、平局、大 token id、TP1/2/4、子组 collective 次序与 payload。
- 通信元数据明确为 `estimate / collective_input_per_rank_per_row`。Qwen3-32B、BF16、TP8 的 37984 B 对 8 B，即 4748×，仅是输入字节比。
- 证据 schema v2 校验真实模型、冻结 workload、协议/容量、逐 step 路径与计数、原始请求、profile artifact、session 时间和 ABBA 实际顺序。
- 零分母输出 null/N/A；证据不全、损坏或 CV 超限是 `incomplete`，不构成候选无效的研究结论。
- NPU preflight 观察至少 1 秒、至少三轮，检查各设备整窗最大 HBM/AICore；运行中继续采样。缺设备、查询出错和未完成采集不能当作空闲。
- Bash runner 对成功、失败与中断保留 session 状态、原始退出码、`EXIT_STATUS`、哈希和归档；拒绝覆盖已有输出、archive 或 checksum sidecar。八类真实 Bash 生命周期测试已通过。

主要文件：`src/minigpt/qwen3_tp.py`、`serving.py`、`decode_critical_path.py`，`scripts/run_v08_decode_vocab_ab.sh`，`benchmarks/check_v08_npu_idle.py`、`summarize_v08_decode_ab.py`。

完整说明：`docs/V0_8_DECODE_CRITICAL_PATH.md`。

## 3. v0.9 完成内容

### 后端与测量入口

`src/minigpt/backends/` 隔离 CPU/CUDA/Ascend 的 runtime、allocator、event、RNG、Profiler 和遥测；模型数学与 KV/scheduler 仍使用普通 PyTorch Tensor。公共 `RuntimeContext` 与分布式接口兼容，`torch_npu` 延迟加载。

两类 Qwen3 benchmark 直接接收相同 `WorkloadTrace` 原始文件。静态入口不覆盖文件中的 GenerationConfig，空 EOS 表示固定长度；请求 id、输入 token、逐请求 TTFT/TPOT/E2E 和完整输出 digest 绑定原始请求。所有 rank 核对完整 256-bit digest。

`memory_measurements` 来自 measured repeats 的 allocator 快照，单位 bytes，逐 rank 用 int64 汇集；独立 profile replay 不污染峰值。CPU 和缺失的可选 reserved counter 为 null/unsupported。各 rank 峰值之和不等于某一时刻整机峰值。

CPU 拓扑记录 0 张 accelerator 卡，进程 rank 不冒充物理设备。设备映射核对实际 visibility、LOCAL_RANK 与 runtime device；CUDA 设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，NPU 明确 logical device / physical card / chip 对应关系。

### 真实 Profiler 与兼容

- CPU/CUDA 使用真实 `torch.profiler`；Ascend 使用真实 `torch_npu.profiler`。
- portable manifest schema v2 绑定全部选中 rank、实际采集窗口、backend、代码提交、workload 文件/语义哈希和 artifacts。
- profile 在 measured repeats 之后独立 replay，重新核对原始输出与 digest。
- continuous benchmark 的 `--profile` 默认仍保留旧 Ascend schema v1；v0.9 加 `--profile-format portable`。旧 `minigpt.ascend_profiling` 由兼容 shim 保留。
- CPU 没有 accelerator compute/communication fraction，字段为 null；CUDA 未观测到通信不能直接当成 0。Ascend Stage 与 Kineto Host step 分母不同，不直接比较其百分比。

### 可恢复真实矩阵

| 预设 | TP / logical devices | cases | 独立进程点 |
|---|---|---:|---:|
| Ascend A3 | 2 / 4 / 8 / 16，每卡 2 chips | 22 | 88 |
| Ascend A5 | 1 / 2 / 4 / 8，每卡 1 device | 22 | 88 |
| CUDA | 1 / 2 / 4 / 8，完整 GPU | 22 | 88 |
| CPU 正确性 | 1 / 2 / 4 个进程，0 accelerator | 8 | 16 |

硬件每 case 使用 ABBA，单进程 1 warmup + 3 measured + 独立 profile；CPU 使用 AB、2 measured。硬件有 short 和 long-prefill 输入。KV 比 recompute/cache，TP 比最小 TP/更大 TP，batching 比相同固定 B 的 static cache/continuous fixed-slot cache。

当前 batching 请求同时到达、固定输出长度，不覆盖持续到达、运行中 refill 或 batch-size 曲线。Prefill/Decode 是 KV/TP 对照的独立阶段指标；不同入口阶段边界不强行相除。launcher 当前为单机多设备；A3 TP16 支持 KV-head replication，不能仅凭 KV heads 少于 TP 拒绝计划。

每 attempt 留存原始命令、环境探测、设备空闲多轮采样、起止时间、退出码、日志、benchmark、profile 和哈希。source、配置、实际解释器/依赖/环境、模型、映射或协议变化会阻断 resume；成功点重新校验后复用，失败/中断/孤儿 attempt 保留。

输出目录有真实进程锁。Linux 使用持锁 subreaper supervisor，跟踪 launcher 与独立 session worker 的 PID/创建标记，在控制器 SIGTERM/SIGKILL 后继续清理和收养孤儿 worker；确认全树停止才释放锁。无法清理时记录 `cleanup_blocked` 并停止整个矩阵，supervisor 异常消失但未确认清理也会由持久登记阻断 resume。Windows 保留进程树终止和创建后异常保护，不宣称拥有 Linux 的 subreaper 保证。前序失败时，同 case 后续点 deferred；恢复不能改变 AB/ABBA 实际顺序。汇总重新读取原始产物，不能只信 journal 的 succeeded。

硬件 preflight 使用 npu-smi / nvidia-smi：至少三轮、至少 1 秒，整窗设备内存占用 ≤10%、计算利用率 ≤5%。这里只是启动前空闲检查；实际真机验收还应结合 trace、稳定性和运行现场审查是否有后续争用。

完整参数、指标口径和产物：`docs/V0_9_BACKENDS_PROFILER_MATRIX.md`。

## 4. 现在只剩的真机工作

1. **v0.8 Atlas A3 TP8**：运行 short/mixed 两个 ABBA，共 8 个独立 session；审查 preflight、运行期遥测、协议/哈希、CV、请求工作量与 profiler，按原门槛得出支持、无端到端收益、回退或 incomplete。
2. **v0.9 A3 / A5 / CUDA**：在相应真实硬件上安装匹配 SDK，核验 device map、拓扑和完整版本字符串，使用真实 Qwen3-32B dense 跑各自 88 点矩阵。目标模型参数量为 `32,762,123,264`。
3. 对已完整的矩阵执行 `--require-complete --require-formal`，审查跨后端 comparable 标记和指标分母；缺设备、OOM、SDK 不支持、失败点或不稳定 case 必须如实保留与补跑。
4. 在真机数值、产物和性能验收通过后，再决定正式 tag 和研究结论；提交/推送证据，保存可验证恢复材料，并做冻结版本独立自查。

需要更多显存的 TP1 点不能用静态容量估算宣布通过；应使用容量足够的真实机器。不能为得到完整矩阵删除失败点、降低模型规模或放松正确性门槛。

## 5. 接机执行命令

v0.8 建议独立目录检出软件检查点：

```bash
git clone --branch upgrade/v0.8-decode-critical-path \
  https://github.com/TuraLucrs/minigpt-train-infra.git minigpt-v08
cd minigpt-v08
git checkout --detach 2a661b1372b521d04c26fefc91cfc6080c72054b
python -m pip install --no-deps -e .
export MODEL_DIR=/path/to/Qwen3-32B
export CANN_VERSION='填写真实完整版本'
export INTERCONNECT_TOPOLOGY='4 physical cards, 2 logical devices per card'
bash scripts/run_v08_decode_vocab_ab.sh
```

v0.8 原始输出为 `runs/v08_decode_vocab_ab/`，以及根目录 `v0.8_decode_vocab_ab_evidence.tar.gz` 和 `.sha256`。失败重新执行必须设置新的 `V08_OUTPUT_ROOT` 与 `V08_ARCHIVE`，不能覆盖历史失败。默认冻结 workload 从 `v0.7_ascend_evidence.tar.gz` 提取；也可用 `WORKLOAD_DIR` 指向原始文件，脚本会核验冻结哈希。

v0.9 使用同一个干净软件提交跑所有后端，避免跨机器源码不同：

```bash
git clone --branch upgrade/v0.9-backends-profiler-matrix \
  https://github.com/TuraLucrs/minigpt-train-infra.git minigpt-v09
cd minigpt-v09
git checkout --detach 5e9efbba4473f54bb6e5c31beead4f4bc263179e
python -m pip install --no-deps -e .
python benchmarks/run_v09_matrix.py --config configs/v09_ascend_a3.json --list-points
export CANN_VERSION='填写真实完整版本'
python benchmarks/run_v09_matrix.py \
  --config configs/v09_ascend_a3.json \
  --model-dir /path/to/Qwen3-32B \
  --device-map /path/to/verified-a3-device-map.json \
  --interconnect-topology '填写实际物理卡、芯片与互连拓扑' \
  --output-dir runs/v09_a3
```

A5/CUDA 分别替换 config、实际 map 和输出目录。示例 map 需要对照现场填写，不能直接当作已验证拓扑。窗口不足可加 `--case kv-short-tp2` 跑完整 case；后续同命令加 `--resume`、去掉 case 继续。`--point` 也要求前序成功。新代码/协议改用新输出目录。

```bash
python benchmarks/summarize_v09_matrix.py \
  --input runs/v09_a3 --input runs/v09_a5 --input runs/v09_cuda \
  --output runs/v09_comparison.json --markdown runs/v09_comparison.md \
  --require-complete --require-formal
```

`--require-complete` 检查各矩阵全部点和 case；`--require-formal` 另外要求真实模型、accelerator 和干净提交。跨后端是否能直接比较还需检查 `comparable_hardware_evidence`，它要求相同模型、源码、精度、SLO、warmup/repeats、profile/capacity 合同和工作量。

## 6. 软件回归与故障定位

Linux CI workflow 为 `.github/workflows/v09-backends-profiler-matrix.yml`，它完整列出 20 个脚本，设置 `MINIGPT_RUN_GLOO_TESTS=1`。可以按该列表复现，不跳过真实失败。

新增六个测试入口：

```bash
python tests/test_backends.py
python tests/test_profiler_backends.py
python tests/test_benchmark_portable.py
python tests/test_backend_telemetry.py
python tests/test_benchmark_entrypoints.py
MINIGPT_RUN_GLOO_TESTS=1 python tests/test_experiment_matrix.py
```

`tests/test_experiment_matrix.py` 默认真实执行 CPU TP1 的 4 点，并正确断言全矩阵仍缺 12 点；环境变量启用后在同一配置上继续 TP2/TP4，最终要求 16/16 完整。失败/timeout/锁、重试与孤儿记录、resume 复用、raw replay 与 protocol/manifest/hash/identity 篡改均有回归。

矩阵排错先读 `matrix_state.json` 对应 attempt、`preflight.log`、`runtime_probe.json`、`device_preflight.json`（硬件）和 `benchmark.log`；再读 output 下原始 benchmark 与 profile。Linux 进程回收问题另查 `.process_guards/` 和 attempt 的 `*.process_tree.json`，根据记录的 PID/创建标记核查现场；删除锁或登记不能证明 worker 已停止。不要直接修改 journal 将失败改成 succeeded。

## 7. 后续边界

当前没有实现在线 TP/Replica/SLO 控制器、Paged KV、Chunked Prefill 或 Speculative Decoding。先完成这两版真机数据和解释，依据证据挑选下一项单变量假设。

v1.0 仍是可复现的推理研究基础设施，最终研究成果需要真实基线、消融或 A/B。SLO-aware TP/Replica/Routing 只能在共同协议布局矩阵、最优配置变化信号和相关工作差距成立后继续，不能先写控制器再寻找理由。训练支线不阻塞推理主线。

## 8. 备份范围与恢复

同批交付提供源码快照、相对 `39448e35a6a5a9b8bb146e46d058692534500f09` 的增量 Git bundle、验证材料、`SHA256SUMS` 与 `DELIVERY_RECEIPT.json`。具体文件名、大小、哈希和最终 ref 以回执为准。

- **源码快照**保留提交内的源码、入口、测试、脚本、配置、文档、精选 artifacts 和三个根目录 evidence 压缩包，v0.8 默认脚本所需的冻结 workload 也在其中；不包含 Python venv、模型权重、`v0.7.1_profiling_detail` 大型原始明细或完整 `.git`。
- **增量 bundle**保留本轮起点之后的 v0.8/v0.9 Git 对象与分支，需要已有起点提交及其历史对象的仓库，不能单独 `git clone` 成完整历史。
- 已执行 bundle prerequisite 验证、SHA-256、源码归档逐文件核验和恢复检查，详情见回执。GitHub 分支是远端恢复链，本地 outputs 是可带走的恢复材料；同一磁盘目录不算异地备份，也没有宣称上传到另一个资料库。
- 原用户的 2026-09-10 紧急全量备份及其 12 卷校验仍在旧交接文档中；本次没有改写该旧文档。若使用旧备份，先按旧哈希恢复基线，再应用本次 bundle。

有基线仓库时恢复增量示例：

```bash
git cat-file -t 39448e35a6a5a9b8bb146e46d058692534500f09
git bundle verify /path/to/minigpt_v08_v09_incremental.bundle
git fetch /path/to/minigpt_v08_v09_incremental.bundle \
  refs/heads/upgrade/v0.8-decode-critical-path:refs/remotes/recovery/v08 \
  refs/heads/upgrade/v0.9-backends-profiler-matrix:refs/remotes/recovery/v09
git switch -c recovered-v09 refs/remotes/recovery/v09
```

若 `cat-file` 或 bundle prerequisite 校验失败，先从 GitHub fetch 起点分支与历史；不要强行跳过 prerequisites。正式实验仍建议从精确 Git 提交检出，以保留 clean commit provenance。

后续继续时先读取 `docs/PROJECT_WORKING_AGREEMENT.md`、路线文档和两版专项说明。保留所有未完成代码和失败证据，遇到中断不 reset/clean，不覆盖当前工作树。
