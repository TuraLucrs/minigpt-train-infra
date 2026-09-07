# v0.3 可测量单设备推理基线

## 1. 版本目标

v0.2.x 的生成只是训练结束后的附带 sample，生成策略、模型执行、设备行为和性能统计混在一起。
v0.3 把它升级为独立且可测量的推理 workload：

```text
prompt 文本
→ tokenizer
→ Prefill
→ 选择首 token
→ Decode 循环
→ completion 文本
```

本版本的重点是建立正确阶段边界和可信基线，不提前实现 KV Cache。Decode 每步重算最多
`block_size` 个上下文 token，Benchmark 名称因此明确记录为
`single_request_recompute_decode`。

## 2. 主要改动

### `src/minigpt/runtime.py`

增加最小 `RuntimeContext`：

- 解析 CPU/CUDA 和 fp32/fp16/bf16；
- 创建 autocast context；
- 集中设备同步、显存统计和设备信息；
- 用 capability 表达 Event、显存统计和 fused AdamW 支持；
- 将训练计时器从日志文件迁移到设备边界。

这不是重新包装 PyTorch，而是把确实随硬件变化的系统操作集中起来，为以后 Ascend 后端留出
窄适配点。

### `src/minigpt/inference.py`

- `GenerationConfig`：定义生成长度、greedy/sample、temperature、top-k 和 seed；
- `GenerationResult`：区分 prompt token、生成 token 和完整 token；
- `MiniGPTModelRunner`：明确 `prefill()` 与 `decode()`；
- `InferenceEngine`：负责 tokenizer、token 选择和生成循环；
- `load_minigpt_engine()`：从 v0.2.x 训练 checkpoint 与 tokenizer 恢复推理模型。

生成编排不再属于 `MiniGPT` 模型本体。模型只负责 `input_ids → logits`，推理引擎负责怎样
反复调用模型并选择 token。这为 v0.5 的真实模型 adapter 避免维护第二套调度逻辑。

### `src/minigpt/benchmark.py`

普通生成不逐 token 添加测量同步。Benchmark 使用独立同步路径定义：

- `prefill_ms`：Prefill 开始到首 token 在 device 上完成；
- `TTFT`：请求开始到首 token 取回并 decode 为文本；
- `TPOT`：后续 token 计算、取回并 decode 为文本的平均同步间隔；
- `E2E latency`：请求开始到完整文本 decode 完成；
- input/output/decode tokens/s；
- 峰值设备内存；
- 每个 Decode step 原始延迟；
- 多次重复的 min/max/mean/median/p50/p90/p99。

Warmup 不进入正式 runs。JSON 同时保存测量定义、环境、请求参数、原始 runs 和 summary，避免
只留下一个无法追溯口径的吞吐数字。`experiment.py` 另外记录 Git commit/dirty 状态、启动
命令、checkpoint 路径和 SHA-256；这部分按应用层黑箱使用，不进入推理算法重点讲解。

### 独立入口

- `infer.py`：加载 checkpoint 并生成文本；
- `benchmarks/infer_single_device.py`：执行 warmup、多次测量并保存 JSON 报告。

### 训练路径变化

- `train.py` 复用 `RuntimeContext`，不再自己处理 CUDA 选择、autocast、Event 和显存；
- `--sample` 复用正式 `InferenceEngine`；
- `model.py` 删除旧的附带 `generate()`，模型与推理编排职责分离；
- 训练数学、optimizer、checkpoint 和日志语义不变。

## 3. 正确性基线

`tests/test_inference.py` 检查：

- `prefill()` 等于完整 forward 的最后位置 logits；
- v0.3 `decode()` 等于重算上下文后的最后位置 logits；
- greedy generation 完全确定；
- sample 使用相同 seed 可复现；
- prompt/generated/all token 边界正确；
- TTFT、TPOT、Decode step 数量和吞吐公式一致；
- warmup 不进入正式 runs；
- checkpoint 与相邻 tokenizer 可以被独立推理入口恢复。

既有三组测试继续覆盖训练核心、reference parity 和精确 resume，防止公共 Runtime 抽取造成
训练回归。

## 4. 2026-08-25 CPU 验收

环境：PyTorch `2.8.0+cpu`，tiny CPU 配置，4 step smoke training。

通过：

```text
tests/test_core.py
tests/test_reference_parity.py
tests/test_resume_consistency.py
tests/test_inference.py
```

端到端检查通过：

```text
4 step training
→ latest.pt + tokenizer.json
→ infer.py 生成 4 token
→ warmup=1、repeats=3 Benchmark
→ inference_benchmark.json
```

一次共享 CPU smoke run 的中位数约为 TTFT `0.542 ms`、TPOT `0.597 ms`、E2E
`2.271 ms`。这些数字只证明测量链路可运行，不能外推为真实模型、CUDA 或 Ascend 性能结论。

GPU FP16/BF16、CUDA Event、显存和 kernel 行为仍需在实际 GPU 环境补测。

## 5. 明确限制

- 只有单请求、batch size 1；
- 没有 EOS 停止，固定生成 `max_new_tokens`；
- 没有 KV Cache，每个 Decode step 重算上下文；
- prompt 超过 `block_size` 时只把最后 `block_size` 个 token 送入模型；
- 字符 tokenizer 和 tiny MiniGPT 只承担 reference/CI，不承担工业性能结论；
- 当前 Runtime 只实现 CPU/CUDA，Ascend 仅保留架构边界；
- 没有真实模型、Tensor Parallel、Continuous Batching 或服务接口。

## 6. v0.4 的唯一主目标

在不改变 v0.3 外部生成语义和指标定义的前提下，实现 KV Cache 与静态 Batching，并用 v0.3
的重算路径验证：

```text
cached logits ≈ uncached logits
cached greedy tokens == uncached greedy tokens
```

然后比较相同 prompt/output 配置下的 TTFT、TPOT、吞吐和峰值显存。只有正确性对齐后，性能
数字才有意义。
