# MiniGPT 推理 Infra

一个从透明的 MiniGPT 训练闭环起步、逐步建设真实大模型推理 Infra 的学习与工程项目。

项目最高原则：

> 训练是前置学习阶段，用来建立模型、优化器、精度、显存、通信等基础认知；推理 Infra
> 才是项目主线、专项研究方向和最终交付成果。

项目先用 tiny model 把系统链路真正跑通、看懂、改得动，再把正式性能工作迁移到真实模型：

- 数据加载与 tokenizer
- GPT / Transformer block
- forward、loss、backward、optimizer
- mixed precision: fp16 / bf16
- gradient accumulation
- learning rate scheduler
- checkpoint 保存与断点续训
- 日志记录：loss、tokens/s、GPU memory
- 单卡训练 baseline
- 独立生成入口和 deterministic greedy baseline
- Prefill/Decode 阶段边界
- TTFT、TPOT、E2E latency、吞吐和设备内存 Benchmark
- 逐层预分配 KV Cache 与只处理新增 token 的 Decode
- 不同 prompt 长度、attention mask、EOS 和静态 batch
- KV Cache 与 recompute 的正确性门和性能对照
- Qwen3 tokenizer/config/safetensors 与真实模型数学
- RoPE、RMSNorm、SwiGLU、GQA 和 Qwen3 KV Cache
- Transformers logits 对齐与 Qwen3-32B HBM/TP 规划
- torchrun process group 与 CPU/Gloo、CUDA/NCCL、Ascend/HCCL 后端边界
- Qwen3 attention/MLP/vocabulary Tensor Parallel 与 rank-local safetensors 加载
- TP rank 0 token 广播、逐 rank HBM、collective 和 Scaling 报告
- Continuous Batching、请求状态机与 fixed-slot KV 生命周期
- open-loop/closed-loop 可重放 workload 与 least-loaded 多副本路由
- 同一 8 devices 上 TP8、2×TP4、4×TP2 的真实并发吞吐比较
- request/s、goodput、队列/TTFT/TPOT/E2E、动态 batch、KV/HBM/AICore 指标

## 当前阶段

`v0.1～v0.7` 已冻结。`v0.6-qwen3-tensor-parallel` 于 2026-09-04 在 Ascend Atlas A3
完成真实 Gloo/HCCL 与 Qwen3-32B TP=2/4/8 验收；精选证据位于
`artifacts/v0.6_qwen3_tp_acceptance/`。`v0.7` 的 A/B/C/D 已经完成：Continuous
Batching、请求/KV slot 生命周期、多副本布局、可重放 workload、服务指标，以及真实
Ascend 8 卡上的 TP8、2×TP4、4×TP2 共 18 组正式验收。完整与精选证据分别位于
`v0.7_ascend_evidence.tar.gz` 和 `artifacts/v0.7_qwen3_continuous_batching_acceptance/`。
正式模型固定为 `Qwen/Qwen3-32B`，tiny 模型只承担白盒正确性与快速 CI。

当前开发分支是 `v0.7.1` Ascend Profiling Gate：在不改写 v0.7 正式结果的前提下，增加
有界的全 rank Profiler 采集、服务阶段范围、关键 artifact 哈希和六点诊断矩阵，为 v0.8
只选择一个有真实瓶颈证据的专项。软件实现与 Atlas 证据是两个门禁；六点真机结果完整前
不冻结 `v0.7.1`。

项目第一阶段没有直接堆叠 DDP、FSDP、DeepSpeed，而是先把**单卡训练系统的完整闭环**吃透：

```text
文本
-> tokenizer
-> token ids
-> x/y batch
-> GPT forward
-> next-token loss
-> backward
-> optimizer step
-> lr schedule
-> log
-> checkpoint
-> resume
```

DDP 可以作为学习多进程、rank、process group 和 collective 的短实验；FSDP/ZeRO 是按需
训练支线。它们都不再阻塞 KV Cache、真实模型、Tensor Parallel 和推理调度。

已完成的 v0.1 目标是：

```text
把单卡 GPT 预训练系统写完整、注释写明白、能跑、能断点续训、能记录指标。
```

## 当前版本与教学基线

Git 标签 `baseline-v0.1` 保存了完整教学实现，其中 LayerNorm、GELU、
cross entropy、AdamW、梯度裁剪和 loss scaling 都是手写版本。

Git 标签 `v0.2-native-single-device` 保存第一批原生算子升级；`v0.2.1-single-device-closeout`
完成单设备训练的可靠性、热路径和回归测试收尾；`v0.2.2-single-device-correctness`
修复 CUDA resume 设备映射并澄清窗口指标口径。

`v0.3-measurable-single-device-inference` 将生成编排移出模型本体，增加独立 `infer.py`、
Prefill/Decode reference runner、最小 Runtime 边界和同步单请求 Benchmark。v0.3 的 Decode
仍重算完整有效上下文，不使用 KV Cache；这正是 v0.4 的对照基线。

`v0.4-kv-cache-static-batching` 增加逐层预分配 K/V、Prefill 写缓存、单 token Decode、
不同有效长度的静态 batch、EOS 停止和每请求独立采样 RNG。`recompute` 路径继续作为 oracle，
任何性能报告都必须先通过生成 token 一致性门禁。

`v0.5-qwen3-real-model` 接入本地 Hugging Face Qwen3 权重和 tokenizer，实现 RoPE、RMSNorm、
SwiGLU、GQA、full/cached forward，并建立 Transformers logits 对齐、32B 精确参数量和静态
显存门禁。32B BF16 单卡 64 GiB 没有可靠运行余量，所以正式性能从 v0.6 TP≥2 开始记录。

`v0.6` 使用一进程一 logical device 的 Tensor Parallel，按列/行切分 attention 与 MLP、按词表
切分 Embedding/LM head，并只从 safetensors 读取本 rank 参数。rank 0 统一选择并广播 token；
Benchmark 记录模型加载、逐 rank HBM、TTFT/TPOT/吞吐、collective 与相对最小可运行 TP
基线的 Scaling Efficiency。真实 Atlas A3 验收显示单 rank HBM 基本按 `1 / TP` 缩减，但
batch=1 单请求没有随 TP 增加而加速；项目不把容量扩展包装成吞吐扩展。

当前工程化分支已经把学完且官方实现更成熟的部分逐步替换为 PyTorch 原生算子。

当前仍显式实现：

- 字符级 tokenizer：`src/minigpt/tokenizer.py`
- GPT 数据 batcher：`src/minigpt/data.py`
- causal multi-head self-attention：`CausalSelfAttention`
- Transformer block：`TransformerBlock`
- cosine warmup learning rate scheduler：`cosine_lr`
- training loop：`train.py`
- checkpoint / resume glue：`train.py` + `checkpoint.py`

已升级为 PyTorch 原生实现：

- LayerNorm：`nn.LayerNorm`
- GELU：`torch.nn.functional.gelu`
- next-token cross entropy：`torch.nn.functional.cross_entropy`
- Q/K/V projection：单个 `nn.Linear(n_embd, 3*n_embd)`
- causal attention kernel：`torch.nn.functional.scaled_dot_product_attention`
- optimizer：`torch.optim.AdamW`
- AdamW 参数组：矩阵/Embedding 权重 decay，bias 与 LayerNorm 参数 no-decay
- gradient clipping：`torch.nn.utils.clip_grad_norm_`
- fp16 loss scaling：`torch.amp.GradScaler`
- 训练 loss 在设备上按窗口累计，只在记录边界取回 CPU
- CUDA event/窗口计时，不再每个 optimizer step 全局同步
- 原子 checkpoint；`latest.pt` 优先硬链接 numbered checkpoint，避免重复序列化

升级必须通过核心测试、精确断点续训测试以及固定配置的 loss/吞吐对照。

## 项目结构

```text
minigpt-train/
  README.md
  requirements.txt
  pyproject.toml
  train.py
  infer.py
  infer_qwen3.py
  infer_qwen3_tp.py
  benchmarks/
    infer_single_device.py
    infer_static_batch.py
    infer_qwen3_single_device.py
    infer_qwen3_static_batch.py
    check_qwen3_parity.py
    plan_qwen3_memory.py
    runtime_smoke.py
    tp_hardware_smoke.py
    infer_qwen3_tp.py
    benchmark_tp_collectives.py
    summarize_tp_scaling.py
    generate_serving_workload.py
    partition_serving_workload.py
    infer_qwen3_continuous_batching.py
    sample_npu_telemetry.py
    summarize_serving_layouts.py
    summarize_v07_acceptance.py
    summarize_v071_profile.py
    summarize_v071_profiling_gate.py
  configs/
    tiny_cpu.json
    tiny_gpu.json
    qwen3_32b_official.json
  data/
    tiny_corpus.txt
  docs/
    INDUSTRIALIZATION_OPTIMIZATION_PLAN.md
    INFERENCE_FIRST_ROADMAP.md
    V0_3_MEASURABLE_INFERENCE.md
    V0_4_KV_CACHE_STATIC_BATCHING.md
    V0_5_QWEN3_REAL_MODEL.md
    V0_6_QWEN3_TENSOR_PARALLEL.md
    V0_7_CONTINUOUS_BATCHING.md
    V0_7_1_ASCEND_PROFILING_GATE.md
    PROJECT_WORKING_AGREEMENT.md
    V0_2_1_SINGLE_DEVICE_CLOSEOUT.md
    PLAN_REVIEW.md
    LEARNING_GUIDE.md
    ROADMAP_DDP_FSDP_DEEPSPEED.md
  scripts/
    run_tiny_cpu.ps1
    run_tiny_gpu.ps1
    resume_latest_cpu.ps1
    create_tiny_qwen3_fixture.py
    run_v071_ascend_profiling_gate.sh
  src/
    minigpt/
      tokenizer.py
      data.py
      model.py
      optim.py
      checkpoint.py
      logging_utils.py
      config.py
      runtime.py
      inference.py
      benchmark.py
      experiment.py
      qwen3.py
      qwen3_inference.py
      distributed.py
      qwen3_tp.py
      memory_planner.py
      serving.py
      workload.py
      replay.py
      replica.py
      serving_benchmark.py
      serving_acceptance.py
      serving_layout.py
      serving_telemetry.py
      ascend_profiling.py
      profiling_gate.py
  tests/
    test_core.py
    test_reference_parity.py
    test_resume_consistency.py
    test_inference.py
    test_kv_cache.py
    test_qwen3.py
    test_qwen3_tp.py
    test_tp_scaling.py
    test_continuous_batching.py
    test_serving_workloads.py
    test_profiling_gate.py
```

## 环境准备

使用 Python 3.10+（与 `pyproject.toml` 一致）。

```powershell
cd F:\ai-infra-projects\minigpt-train
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你已经有 PyTorch 环境，也可以直接运行测试。

## 先跑测试

```powershell
cd F:\ai-infra-projects\minigpt-train
python tests/test_core.py
python tests/test_reference_parity.py
python tests/test_resume_consistency.py
python tests/test_inference.py
python tests/test_kv_cache.py
python tests/test_qwen3.py
python tests/test_qwen3_tp.py
python tests/test_tp_scaling.py
python tests/test_continuous_batching.py
python tests/test_serving_workloads.py
python tests/test_profiling_gate.py
```

看到：

```text
All core smoke tests passed.
Reference-vs-optimized parity tests passed.
Exact resume consistency test passed.
v0.3 inference and benchmark tests passed.
v0.4 KV Cache and static batching tests passed.
Qwen3 parity, cache, loading and memory planning tests passed.
v0.6 Qwen3 Tensor Parallel simulation tests passed.
v0.6 TP scaling summary tests passed.
v0.7 continuous batching scheduler tests passed.
v0.7 workload replay and multi-replica routing tests passed.
v0.7.1 profiling manifest and decision-gate tests passed.
```

第一项检查 tokenizer、model、optimizer 参数组、checkpoint 原子保存和旧版本迁移；
第二项对照教学公式与原生 LayerNorm、GELU、cross entropy、SDPA 的输出和梯度；
第三项检查连续训练和从中间 checkpoint 恢复能否得到完全一致的最终训练状态。
第四项检查 Prefill/Decode、greedy/sample、独立 checkpoint 加载和推理指标口径。
第五项检查 cached/recompute logits 与生成一致性、缓存原地复用、不同长度 mask、窗口滚动、
EOS 和静态 batch Benchmark。
第六项检查 Qwen3/Transformers logits、cached Prefill/Decode、单/分片 safetensors、32B
参数量与 HBM 规划。
第七项用多个 rank-local 模型和确定性线程 collective 检查 TP=2/4 的参数分片、full/cached
logits、KV-head 复制和生成一致性；设置 `MINIGPT_RUN_GLOO_TESTS=1` 后额外执行真实 Gloo
process group。第八项检查报告可比性、speedup 与 Scaling Efficiency 公式。
第九项检查请求状态、动态加入/退出、fixed-slot KV、EOS、取消、背压、失败清理和逐请求 RNG。
第十项检查可重放 workload、open/closed loop、路由、真实并发布局汇总、遥测与证据降级门禁。

## Qwen3 v0.5 快速验收

先创建离线 tiny fixture；它只用于正确性，不能作为性能数据：

```powershell
python scripts/create_tiny_qwen3_fixture.py
python benchmarks/check_qwen3_parity.py --model-dir runs/tiny_qwen3_fixture
python infer_qwen3.py `
  --model-dir runs/tiny_qwen3_fixture `
  --device cpu --precision fp32 --max-new-tokens 4 `
  --prompt "Hello world"
python benchmarks/infer_qwen3_static_batch.py `
  --model-dir runs/tiny_qwen3_fixture `
  --device cpu --precision fp32 --max-new-tokens 4 `
  --prompt "Hello world" --prompt "Qwen inference"
```

正式 32B 运行前先做容量和后端门禁：

```powershell
python benchmarks/plan_qwen3_memory.py --tp 1 2 4 8 16
python benchmarks/runtime_smoke.py --device npu --precision bf16
```

静态估算显示：32B BF16 权重约 62,488.8 MiB；加最小 KV Cache、workspace 和 runtime
reserve 后约 65,592.8 MiB，超过 64 GiB。因此 v0.5 不拿单卡 32B 冒险做正式性能结论；
v0.6 完成 TP 分片加载后，从 TP=2/4/8 开始在机器上留正式记录。详细边界见
`docs/V0_5_QWEN3_REAL_MODEL.md`。

## Qwen3 v0.6 Tensor Parallel 硬件验收

软件 smoke 可以先在 tiny fixture 上执行：

```bash
python tests/test_qwen3_tp.py
python tests/test_tp_scaling.py
python infer_qwen3_tp.py \
  --model-dir runs/tiny_qwen3_fixture \
  --device cpu --precision fp32 --max-new-tokens 4 --prompt "Hello world"
```

允许本地 TCP 的 Linux 环境还必须执行：

```bash
MINIGPT_RUN_GLOO_TESTS=1 python tests/test_qwen3_tp.py
```

Ascend 上在加载 32B 权重前，必须先让 tiny Qwen3 经过真实 HCCL、分片 loader、TP
AllReduce/AllGather、KV Cache 和 token broadcast：

```bash
python benchmarks/runtime_smoke.py --device npu --precision bf16
torchrun --standalone --nproc-per-node=2 benchmarks/tp_hardware_smoke.py \
  --device npu --backend hccl --precision bf16 \
  --output runs/tp2_hardware_smoke.json
```

显式指定 NPU/CUDA 和低精度的 smoke 在设备或精度不可用时直接失败，不会回退 CPU/FP32
后输出 `passed`。
真实 32B 使用 `torchrun` 启动 TP=2/4/8，先过对应规模的 hardware smoke 和 collective，
再跑相同 workload 的推理报告，最后合并 Scaling。完整命令、拓扑字段和证据等级见
[`docs/V0_6_QWEN3_TENSOR_PARALLEL.md`](docs/V0_6_QWEN3_TENSOR_PARALLEL.md)。

## v0.7 Continuous Batching 与多副本吞吐

软件正确性先执行：

```bash
python tests/test_continuous_batching.py
python tests/test_serving_workloads.py
```

v0.7 在 v0.6 TP runner 上增加请求状态机、动态 admission、fixed-slot KV 分配/释放、
rank 0 调度计划、可重放 open/closed-loop workload 和 least-loaded 多副本路由。真实 8 卡
比较始终使用一个 8 进程作业，通过 TP subgroup 形成 TP8、2×TP4 或 4×TP2，布局吞吐取各
replica 的真实并发墙钟区间，不把单副本吞吐乘副本数。正式对比还固定相同的全局 slot、
waiting queue 和 `max_seq_len`，避免把额外调度容量误算成多副本布局收益。

正式汇总同时检查原始 workload、报告 manifest、模型/权重/提交、硬件/软件、SLO、请求集合、
完整输出 token digest、启动偏差和每卡 AICore 遥测，并重新判定 replica 候选资格、重算逐请求
`slo_met`；最终验收会从这些原始 artifacts 重算整份三布局 comparison，不能靠修改标签或
布尔门禁伪造正式结果。未提供遥测或任一证据不完整
时仍输出开发报告，但不会标为正式性能结论。架构、指标、完整测试和 Ascend 运行命令见
[`docs/V0_7_CONTINUOUS_BATCHING.md`](docs/V0_7_CONTINUOUS_BATCHING.md)。

## v0.7.1 Ascend Profiling Gate

v0.7.1 不实现某个性能优化，而是先补上 v0.8 的选题证据。普通 warmup/三次 measured replay
结束后，runner reset 并复用同一 admission script 做一次独立 profile replay；后者不进入
TTFT、TPOT、E2E、吞吐或 goodput 聚合，并必须生成与 measured replay 相同的完整输出 digest。

正式 Atlas gate 固定 8 个 rank、Level1 和 PipeUtilization，并使用按问题定向的有界窗口：
short 为 8/2/4，long-prefill 为 6/1/4，mixed 为 14/1/4。门禁不仅要求每个 rank 的 operator、kernel、
step trace、timeline 和 communication artifact 通过哈希校验，还验证窗口实际覆盖目标阶段。
三组问题各比较两个 layout，共六个 8 进程作业；只有六点全部完整时才允许据此选择一个
v0.8 A/B 研究方向。协议、命令和冻结标准见
[`docs/V0_7_1_ASCEND_PROFILING_GATE.md`](docs/V0_7_1_ASCEND_PROFILING_GATE.md)。

## 独立运行一次推理

先使用已有训练产物：

```powershell
python infer.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --max-new-tokens 32 `
  --strategy greedy `
  --device cpu
```

默认使用 `--decode-mode kv_cache`。需要运行 v0.3 对照路径时传入
`--decode-mode recompute`。

`greedy` 每次选择 logits 最大的 token，适合建立稳定正确性基线。需要观察随机采样时再使用：

```powershell
python infer.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --max-new-tokens 32 `
  --strategy sample `
  --temperature 0.9 `
  --top-k 20 `
  --seed 1337
```

## 运行单设备推理 Benchmark

```powershell
python benchmarks/infer_single_device.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --max-new-tokens 32 `
  --device cpu `
  --decode-mode compare `
  --warmup 2 `
  --repeats 5 `
  --output runs/inference_benchmark.json
```

`compare` 会分别运行 recompute 和 KV Cache，先验证生成 token 完全一致，再计算 TPOT 加速比
与峰值内存差。报告保留每次原始结果，并汇总 median、p50、p90、p99。指标定义为：

- `prefill_ms`：Prefill 开始到首 token 在 device 上计算完成；
- `TTFT`：请求开始到首 token 已经取回并 decode 为文本；
- `TPOT`：除首 token 外，后续 token 计算、取回并 decode 为文本的平均间隔；
- `E2E latency`：请求开始到完整结果 decode 完成；
- `output tokens/s`：生成 token 数除以端到端时间；
- `peak device memory`：模型已经加载后的请求测量窗口内，模型与推理共同占用的峰值设备内存。

CPU 上的 tiny 结果只用于正确性和本机回归，不代表真实模型或工业硬件性能。

静态 batch 使用多次 `--prompt` 指定同一批请求：

```powershell
python benchmarks/infer_static_batch.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --prompt "GPT" `
  --max-new-tokens 32 `
  --device cpu
```

## 跑一个 CPU 小训练

```powershell
cd F:\ai-infra-projects\minigpt-train
python train.py --config configs/tiny_cpu.json --max_steps 10 --sample --overwrite
```

或者：

```powershell
.\scripts\run_tiny_cpu.ps1
```

训练输出会类似：

```text
step     1 | train loss 3.80 | lr 2.00e-04 | 12000 tok/s | gpu 0.0/0.0 MB | grad 1.23
```

输出目录：

```text
runs/tiny_cpu/
  tokenizer.json
  train_log.csv
  latest.pt
  checkpoints/
    step_000010.pt
```

## 输出目录规则

为了避免新实验和旧 checkpoint 混在一起，fresh run 如果发现 `out_dir` 里已经有
`train_log.csv`、`latest.pt`、`tokenizer.json` 或 `checkpoints/`，会直接报错。

你有三个选择：

```powershell
# 继续旧实验
python train.py --config configs/tiny_cpu.json --resume runs/tiny_cpu/latest.pt --max_steps 20

# 新开一个实验目录
python train.py --config configs/tiny_cpu.json --out_dir runs/exp_lr_1e_3

# 明确覆盖旧实验产物
python train.py --config configs/tiny_cpu.json --overwrite
```

resume 时，代码会优先加载 checkpoint 所属 run 目录里的 `tokenizer.json`，并检查
`tokenizer_hash` / `vocab_size`。这可以避免 token id 映射错位。

## 跑一个 GPU 小训练

如果你有 NVIDIA GPU：

```powershell
cd F:\ai-infra-projects\minigpt-train
python train.py --config configs/tiny_gpu.json --sample --overwrite
```

默认使用 `bf16`。如果显卡不支持 bf16，可以改：

```powershell
python train.py --config configs/tiny_gpu.json --precision fp16 --overwrite
```

或者直接用 fp32：

```powershell
python train.py --config configs/tiny_gpu.json --precision fp32 --overwrite
```

## 断点续训

先跑一次训练，生成：

```text
runs/tiny_cpu/latest.pt
```

然后：

```powershell
python train.py --config configs/tiny_cpu.json --resume runs/tiny_cpu/latest.pt --max_steps 20
```

注意：`--max_steps 20` 表示训练到第 20 个 optimizer step，不是再训练 20 step。

## 你应该怎么读代码？

机器实验窗口结束后，学习 v0.5～v0.6 新增内容时推荐顺序：

1. `src/minigpt/runtime.py`
2. `src/minigpt/qwen3.py` 的 config 与完整 forward
3. Qwen3 attention、RoPE、GQA、Prefill/Decode cache
4. `src/minigpt/qwen3_inference.py`
5. `src/minigpt/inference.py` 的通用 `InferenceEngine`
6. `infer_qwen3.py`
7. `src/minigpt/benchmark.py`
8. `benchmarks/check_qwen3_parity.py`
9. `benchmarks/infer_qwen3_single_device.py`
10. `tests/test_qwen3.py`
11. `src/minigpt/distributed.py`
12. `src/minigpt/qwen3_tp.py`
13. `infer_qwen3_tp.py` 与 `benchmarks/infer_qwen3_tp.py`
14. `benchmarks/benchmark_tp_collectives.py` 与 `summarize_tp_scaling.py`
15. `tests/test_qwen3_tp.py` 与 `tests/test_tp_scaling.py`
16. `src/minigpt/serving.py`
17. `tests/test_continuous_batching.py`
18. `src/minigpt/workload.py` 与 `src/minigpt/replay.py`
19. `src/minigpt/replica.py`
20. `src/minigpt/distributed.py` 的 `TensorParallelReplicaContext`
21. `benchmarks/infer_qwen3_continuous_batching.py`
22. `src/minigpt/serving_layout.py` 与 `serving_telemetry.py`
23. `tests/test_serving_workloads.py`
24. `src/minigpt/ascend_profiling.py` 与 `profiling_gate.py`
25. `benchmarks/summarize_v071_profile.py` 与 `summarize_v071_profiling_gate.py`
26. `tests/test_profiling_gate.py`

核心顺序是先看真实模型怎样得到正确 logits 和 cache，再看通用生成与测量；不要先从 CLI
参数或模型目录路径等工程胶水开始。当前机器窗口优先实现与实测，完整逐段讲解暂时后移，
但每次 commit/tag 后的冻结版本自查不会省略。

## 你可以做的实验

先从很小的改动开始：

```powershell
python train.py --config configs/tiny_cpu.json --max_steps 5 --out_dir runs/exp_5_steps
python train.py --config configs/tiny_cpu.json --max_steps 50 --out_dir runs/exp_50_steps
python train.py --config configs/tiny_cpu.json --max_steps 50 --out_dir runs/exp_50_steps_lr_test
```

然后改 `configs/tiny_cpu.json`：

- `batch_size`: 8 改 16，看 tokens/s 变化
- `block_size`: 64 改 32，看训练速度和 loss 变化
- `n_layer`: 2 改 4，看参数量和速度变化
- `learning_rate`: 0.001 改 0.01，看 loss 是否不稳定
- `gradient_accumulation_steps`: 2 改 4，看等效 batch 变大后有什么变化
- `precision`: GPU 上试 `fp32` / `fp16` / `bf16`

每次只改一个参数。这样你才知道变化来自哪里。

## 日志字段怎么看？

`runs/.../train_log.csv` 里有这些字段：

- `step`: 训练循环执行次数（包含因 fp16 溢出而未更新参数的尝试）
- `optimizer_step`: AdamW 实际成功完成的参数更新次数
- `split`: `train` 或 `val`
- `loss`: next-token prediction loss
- `lr`: 当前学习率
- `tokens_per_sec`: 当前纯训练计时窗口内的 wall-time tokens/s，不包含随后执行的 eval/checkpoint
- `gpu_mem_mb`: 当前 GPU 显存占用
- `gpu_peak_mb`: 当前计时窗口的峰值 GPU 显存
- `grad_norm_last`: 窗口最后一步在 gradient clipping 前的梯度总 norm
- `grad_norm_max`: 整个窗口在 gradient clipping 前的最大梯度总 norm
- `loss_scale`: fp16 时的 loss scale
- `skipped_steps`: 当前窗口因 fp16 梯度出现 inf/nan 而跳过的更新次数

## 为什么 tiny_corpus 这么小？

这是学习项目，不是模型效果项目。

小语料的好处：

- CPU 也能跑；
- 每一步都快；
- checkpoint 小；
- 你可以大胆改代码；
- 出错后容易定位。

等你读懂训练链路后，再换更大的文本数据。

## 后续怎么扩展？

当前权威路线（每个版本验收、冻结并讲完变化后，才进入下一版本）：

```text
baseline-v0.1     教学单设备训练闭环
v0.2.x            优化并修正单设备训练
v0.3              可测量的 MiniGPT 单设备推理基线
v0.4              KV Cache、Prefill/Decode 独立路径与静态 Batching
v0.5              Qwen3 真实模型接入、KV Cache 与数值门禁
v0.6              分布式基础短实验与 Tensor Parallel 推理
v0.7              Continuous Batching、调度与 KV 生命周期
v0.7.1            Ascend Profiling Gate 与 v0.8 选题证据
v0.8              基于 Profiling Gate 只选择一个推理专项研究
v0.9              扩展 Ascend 适配、Profiler 后端化与真实多机/多卡 Scaling
v1.0              完整推理 Infra 交付
```

MiniGPT 长期作为白盒 reference、正确性 oracle 和 CPU CI；接入真实模型后，正式性能、显存、
Scaling 和最终报告全部以真实模型为准。完整分工与验收边界见
`docs/INFERENCE_FIRST_ROADMAP.md`。

## 当前版本边界

v0.5 当前故意不做：

- Qwen3 sliding-window attention 或 rope scaling
- 32B 单卡 64 GiB 的无余量强行加载
- Continuous Batching
- DDP/FSDP/ZeRO 完整训练系统
- TensorBoard / WandB
- Paged KV Cache
- Tensor Parallel
- 外部 FlashAttention 或自定义 fused kernel

这些功能按推理主线逐版本进入，不能与真实模型、TP、调度同时改写，否则结果出错或性能变化
时无法归因。版本施工与备份硬约束见 `docs/PROJECT_WORKING_AGREEMENT.md`。
