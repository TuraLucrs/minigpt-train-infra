# v0.6：Qwen3 Tensor Parallel 与多卡 Scaling

## 1. 版本目标与完成边界

v0.6 把 v0.5 的 Qwen3 full/cached inference 从单进程完整权重扩展为单机多进程 Tensor
Parallel。正式模型仍固定为 `Qwen/Qwen3-32B` dense；tiny fixture 只做数学、分片和 CI，正式
结果必须来自真实 32B 权重及 CUDA/NCCL 或 Ascend/HCCL。

本版本的软件实现完成后进入硬件验收阶段。只有以下条件全部满足，才能创建最终 v0.6 tag：

- tiny TP=1/2/4 full logits、cached Prefill/Decode、greedy generation 一致；
- Linux 环境的 TP=2 Gloo process-group 门禁通过；
- 真实 32B 至少完成 TP=2/4/8 HCCL 或 NCCL 相同 workload 的报告；
- 每份正式报告包含完整权重 SHA-256、干净 Git commit 和物理/逻辑设备拓扑；
- Scaling 汇总证明所有报告来自同一模型、同一请求和同一提交；
- 完成提交后的冻结版本独立自查，再决定高性价比改进是否进入 v0.7。

## 2. 进程、设备与通信边界

每个 `torchrun` rank 对应一个 logical device：

```text
torchrun
  ├─ rank 0 / local_rank 0 / device 0
  ├─ rank 1 / local_rank 1 / device 1
  └─ ...
```

`DistributedContext` 在初始化 process group 前绑定 local device，并按设备选择：

| device | backend |
|---|---|
| CPU | Gloo |
| CUDA | NCCL |
| NPU | HCCL |

显式请求 `cuda` 或 `npu` 时不允许静默回退 CPU；已有 process group 的 rank、world size 和
backend 必须与请求完全一致。`world_size=1` 保留无通信路径，便于 tiny smoke 和 TP=1 基线。

## 3. Qwen3 参数分片

设 TP world size 为 `P`。Qwen3-32B 的关键维度是：

```text
query heads       = 64
KV heads          = 8
intermediate size = 25600
vocab size        = 151936
```

分片方式：

| 参数/输出 | 分片维度 | collective |
|---|---|---|
| Embedding | vocabulary rows | hidden state AllReduce |
| Q/K/V projection | output rows | 无 |
| O projection | input columns | output AllReduce |
| gate/up projection | output rows | 无 |
| down projection | input columns | output AllReduce |
| LM head | vocabulary rows | logits AllGather |
| RMSNorm | replicated | 无 |

Q、FFN 和 vocabulary 要求可以被 `P` 整除。`P <= 8` 时 KV heads 正常分片；`P > 8`
时，同一个 KV head 会复制到负责它对应 query group 的多个 rank。以 TP=16 为例，每个 rank
负责 4 个 query heads，相邻两个 rank 共享同一个 KV head 分片。KV Cache 只保存当前 rank
实际需要的 KV heads。

## 4. 分片权重加载

`load_tp_qwen3_from_pretrained()` 先在 meta device 建立当前 rank 的局部模块，再通过
`safetensors.safe_open(...).get_slice()` 读取本 rank 所需行或列。它不会先在每个 rank 构造
一个完整 32B 模型再切片。

加载门禁包括：

- config 必须属于当前已实现的 Qwen3 dense 范围；
- checkpoint 参数集合必须与预期集合完全相同；
- index 中的 shard 映射必须与 shard 实际内容一致；
- 每个 source shape 和 rank-local target shape 必须匹配；
- shard 文件必须直接位于指定模型目录，拒绝路径逃逸。

虽然每个 rank 只把局部 Tensor 搬到 device，第一版 loader 仍会让每个 rank 打开所有 shard。
加载时间和存储并发因此需要实测，不能仅凭局部参数量推断。

## 5. token 选择不能由各 rank 各自决定

模型 collective 后每个 rank 拥有相同的完整 logits，但随机采样仍不应依赖“各 rank 恰好使用
相同 RNG 状态”。`TensorParallelInferenceEngine` 只让 rank 0 执行 greedy/sample token
选择，再广播 `[B,1]` token ids。所有 rank 收到同一 token 后才进入下一次 Decode，避免某个
rank 分叉后在不同计算图位置进入 collective 并最终死锁。

## 6. 正确性门禁

当前环境的默认测试使用线程内确定性 collective 仿真，不需要 socket，同时真正执行多个
rank-local 模型并在每层合并结果：

```bash
python tests/test_qwen3_tp.py
python tests/test_tp_scaling.py
```

它覆盖：

- 官方 32B TP=1/2/4/8/16 plan；
- TP=4 大于 tiny KV-head 数时的复制分支；
- rank-local safetensors 行/列切片；
- embedding/O/down 的 AllReduce 和 LM-head AllGather；
- full logits、cached Prefill、cached Decode；
- 不同 prompt 长度的静态 batch greedy generation；
- scaling 公式和不可比报告拒绝逻辑。

在允许本地 TCP socket 的 Linux 环境再执行真实 Gloo process group：

```bash
MINIGPT_RUN_GLOO_TESTS=1 python tests/test_qwen3_tp.py
```

当前 Codex 容器的 Gloo transport 被平台权限拒绝，错误发生在 process group 建立阶段；因此
线程仿真通过不能冒充 Gloo/HCCL 已通过。

## 7. 正式机器运行顺序

以下命令中的模型目录、物理卡数、每卡芯片数和互联描述必须替换为机器真实值。

先做 Runtime 与 HCCL smoke：

```bash
python benchmarks/runtime_smoke.py --device npu --precision bf16

torchrun --standalone --nproc-per-node=2 infer_qwen3_tp.py \
  --model-dir /path/to/Qwen3-32B \
  --device npu --backend hccl --precision bf16 \
  --chat-template --max-new-tokens 8
```

再测 collective：

```bash
torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_tp_collectives.py \
  --device npu --backend hccl --precision bf16 \
  --message-mb 0.25 --message-mb 1 --message-mb 4 \
  --physical-card-count <physical> --chips-per-card <chips> \
  --interconnect-topology "<真实拓扑>" \
  --output runs/tp2_collectives.json
```

然后以完全相同的 prompt、生成参数、warmup 和 repeats 分别执行 TP=2/4/8：

```bash
torchrun --standalone --nproc-per-node=2 benchmarks/infer_qwen3_tp.py \
  --model-dir /path/to/Qwen3-32B \
  --device npu --backend hccl --precision bf16 \
  --prompt "你好，请介绍一下你自己。" \
  --chat-template --max-new-tokens 32 --warmup 2 --repeats 5 \
  --hash-weights \
  --physical-card-count <physical> --chips-per-card <chips> \
  --interconnect-topology "<真实拓扑>" \
  --run-label qwen3-32b-tp2 \
  --output runs/qwen3_32b_tp2.json
```

把 `--nproc-per-node`、拓扑字段、run label 和输出文件分别改成 TP=4、TP=8。最后汇总：

```bash
python benchmarks/summarize_tp_scaling.py \
  --report runs/qwen3_32b_tp2.json \
  --report runs/qwen3_32b_tp4.json \
  --report runs/qwen3_32b_tp8.json \
  --output runs/qwen3_32b_tp_scaling.json
```

32B BF16 的 TP=1 静态容量已经在 v0.5 判定为没有可靠余量，因此正式 32B Scaling 以 TP=2
为最小可运行基线；TP=1 只在 tiny correctness 中保留。汇总报告会明确写出
`baseline_world_size=2`，不会把 TP=2→8 的结果误写成相对单卡的 8 倍扩展。

## 8. 指标与证据等级

TP 报告在 v0.5 指标之外增加：

- 每个 rank 的模型加载耗时；
- 模型加载完成后的 allocated HBM；
- 每个 rank 的请求峰值 HBM；
- 最大 rank HBM 与所有 rank HBM 合计；
- full/local parameter count 与实际参数占比；
- rank 间生成结果摘要一致性；
- physical card、logical device、chips/card 和 interconnect topology；
- collective 消息大小、延迟、输入带宽和估算 rank 流量。

只有真实 32B、TP≥2、accelerator、完整拓扑、完整权重哈希、干净 Git commit 同时满足时，单次
报告才标为 `formal_qwen3_32b_tp_hashed`。只有所有输入报告还使用同一权重清单、同一 commit、
同一模型配置和同一请求时，汇总才标为 `formal_qwen3_32b_tp_scaling`。

## 9. 当前有意保留的边界

- v0.6 是单机 TP；多机拓扑、容错和 elastic 不在本版本；
- 每步仍 AllGather 完整 vocabulary logits，尚未做 distributed top-k/sample；
- collective 没有与 GEMM 重叠，也没有使用厂商融合算子；
- 静态 batch 仍要求一批请求同时开始，动态加入/退出和 KV 生命周期属于 v0.7；
- 不通过人为预分配无用 Tensor 追求“显存接近 64 GiB”。显存利用率必须来自真实权重、KV、
  batch/context workload；较高 TP 会降低每卡权重占用，v0.7 再通过真实并发提高有效利用率；
- Ascend Profiler 深度归因和 CUDA/NPU 系统对照属于 v0.9，但 v0.6 必须先留下原始 HCCL、
  TTFT、TPOT、吞吐和 HBM 数据。
