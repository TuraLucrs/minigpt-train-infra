# v0.5 冻结前恢复清单（2026-09-03）

## 当前 Git 状态

- 分支：`upgrade/v0.5-qwen3-real-model`
- 基线提交：`a09e5a96879f4a1c985732d9dbefc32e2cfd5b5f`
- 基线标签：`v0.4-kv-cache-static-batching`
- v0.5 尚未 commit/tag，必须保留全部 tracked/untracked dirty state，禁止 reset、clean 或回退。
- GitHub remote 已获用户明确授权；上次 HTTPS push 因缺少用户名/token 失败。v0.5 冻结后必须
  立即重试并读取远端 refs 核验，不能把本地 commit 当作已经上传。

## 已完成实现

- 本地 Qwen3 config、tokenizer、单文件/分片 safetensors 严格加载；
- meta-device 建模与逐参数 CPU→目标设备复制；
- Qwen3 RMSNorm、FP32 RoPE、SwiGLU、GQA、full forward；
- 逐层 KV Cache、Prefill、单 token Decode、左右 padding、静态 batch；
- LM head 前只选择最后有效 hidden state，KV 容量按请求实际长度分配；
- Qwen3 官方多 EOS 停止条件、top-k/top-p sampling；
- Transformers full logits、cached Prefill、cached Decode 数值门禁；
- CPU/CUDA/NPU Runtime 边界、同步、Event、显存和能力元数据；
- 精确 32B 参数量和考虑复制参数/KV heads 的 TP 静态 HBM 规划；
- 单请求、静态 batch、parity、memory planner、Runtime smoke CLI；
- 报告记录代码状态、命令、config/tokenizer/generation config 和可选权重哈希；
- 正式证据等级阻止 tiny/unhashed 结果冒充 32B 正式性能。

## 冻结前独立审查已经发现并修复

- 官方 `generation_config.json` 实际有 `151645`、`151643` 两个停止 token；
- `prompt + max_new_tokens` 的 KV 容量存在一位保守误差；
- Qwen3 KV Cache 未初始化区域可能以 NaN 污染变长 batch 的注意力；
- NPU 缺少能力探测 API 时不应乐观假设支持 BF16；
- 正式报告还应哈希 tokenizer 与 generation config，而不只哈希权重/config；
- TP 显存不能简单使用总参数量除以 TP：norm、KV heads 与取整会产生复制开销。

## 已通过验收

- `compileall`、`git diff --check`；
- 全部 v0.1～v0.5 测试；
- 所有 CLI `--help`；
- tiny Qwen3 fixture 创建、HF parity、生成、单请求 Benchmark、静态 batch Benchmark；
- 官方 32B 参数量与 TP=1/2/4/8/16 静态规划；
- CPU Runtime smoke。

当前环境没有 Ascend NPU，NPU Runtime 只完成官方 API 核验；实际 NPU smoke 必须在用户机器运行，
不得提前声称硬件验证通过。

## 恢复后严格续作顺序

1. 再跑一次针对最新修改的 tiny Qwen3 端到端验收；
2. 更新并异地保存 dirty tar、binary patch、状态/续作清单；
3. 检查 Git diff/status，commit v0.5 并创建 annotated tag；
4. 立即 push 分支和 tag，读取远端 refs 核验；凭据缺失时明确记录失败；
5. 创建、校验并持久保存 v0.5 Git bundle 与源码快照；
6. 从冻结提交的 clean clone 再跑独立自查；重要问题用新 patch commit/tag 修复，不改写已冻结 tag；
7. 把高性价比改进纳入 v0.6，再实现 Qwen3-32B Tensor Parallel 分片加载和正式硬件实验。

## 实验边界

- 正式模型固定为 `Qwen/Qwen3-32B` dense；tiny 只用于正确性与 CLI smoke。
- 32B BF16 TP=1 静态总量约 `65,592.8 MiB`，已超过 `65,536 MiB`，不强行单卡加载。
- 正式 TTFT、TPOT、吞吐、HBM、Scaling 从 v0.6 TP>=2 开始记录。
