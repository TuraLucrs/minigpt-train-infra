# v0.6 Qwen3-32B Tensor Parallel 昇腾验收证据

本目录保存 `upgrade/v0.6-qwen3-tensor-parallel` 在真实 Ascend Atlas A3 上的精选、可直接审阅
证据。原始 ZIP、完整 Git bundle 和分片 bundle 另行离线归档，不重复写入 Git 历史。

## 验收对象

| 项目 | 值 |
|---|---|
| 日期 | 2026-09-04（UTC+8） |
| 最终实机提交 | `ea672a6f31c06fb5cad6ee474203a695b6d2bab3` |
| 模型 | `Qwen/Qwen3-32B` dense，32,762,123,264 参数 |
| 权重 | BF16 safetensors，17 个分片；每份正式报告内嵌完整 SHA-256 清单 |
| 设备 | Ascend Atlas A3，单物理卡 2 个 logical device，每芯 64 GB HBM |
| 软件栈 | Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post5.dev20260821、Transformers 5.14.1 |
| 分布式后端 | HCCL |

## 门禁结果

| 门禁 | 结果 | 证据 |
|---|---|---|
| CPU 回归与真实 TP=2 Gloo process group | 通过 | `console_logs/tests.log` |
| 显式 NPU/BF16 Runtime smoke | 通过，无 CPU/FP32 回退 | `smoke_and_collectives/runtime_smoke_npu.json` |
| tiny Qwen3 HCCL TP=2/4/8 | 全部通过 | `smoke_and_collectives/tp*_hardware_smoke.json` |
| HCCL AllReduce/AllGather TP=2/4/8 | 全部完成 | `smoke_and_collectives/tp*_collectives.json` |
| Qwen3-32B TP=2 短生成 | 通过 | `console_logs/tp2_32b_short.log` |
| Qwen3-32B TP=2/4/8 同 workload | 全部完成，生成 token 一致 | `formal_benchmarks/qwen3_32b_tp*.json` |
| 跨规模可比性门禁 | 通过 | `formal_benchmarks/qwen3_32b_tp_scaling.json` |

首次 32B 短生成暴露 Transformers 5.x `apply_chat_template()` 返回 `BatchEncoding` 的兼容问题；
提交 `ea672a6f` 将结果统一归一化为单行 `list[int]`，随后短生成及正式报告均通过。

## 正式结果

以下均为 2 次 warmup 后 5 次正式运行的中位数，使用同一 prompt、权重、提交和 greedy
生成配置。

| TP | TTFT ms | TPOT ms | E2E ms | 输出 tok/s | 单 rank 峰值 HBM | 相对 TP=2 扩展效率 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 162.1 | 172.1 | 5,494 | 5.824 | 31,462 MB | 1.000 |
| 4 | 188.9 | 174.2 | 5,602 | 5.712 | 15,643 MB | 0.490 |
| 8 | 198.0 | 178.1 | 5,713 | 5.601 | 8,070 MB | 0.240 |

这组结果证明的是：

- rank-local 权重与 KV Cache 能够在真实 NPU/HCCL 上正确运行；
- 局部参数量和单 rank HBM 基本按 `1 / TP` 下降；
- TP=2/4/8 产生完全一致的 greedy token 序列；
- 当前 batch=1、短 prompt 的自回归 Decode 由逐层通信主导，增加 TP 不会带来单请求加速。

不能从这组结果声称：

- TP=8 比 TP=2 吞吐更高；
- 已达到生产并发吞吐或极限 HBM 利用率；
- 该单 prompt、32-token 结果可以代表其他模型或服务负载。

提高有效利用率需要在后续版本引入真实并发、请求调度和 KV Cache 生命周期管理，而不是预分配
无用 Tensor 人为填满显存。

## 已知环境警告

- HCCL watchdog timeout 小于 HCCL execution timeout；本次运行未超时，但正式长任务应统一配置。
- `barrier()` 未显式传 `device_id`，只产生提示；collective 报告均完成。
- CANN/driver 组合触发 allocator 32-byte padding 提示，不影响本次正确性。
- 环境快照记录的物理卡 6 健康告警未参与本次 logical device 0–7 的 TP 验收。

完整命令、端口冲突处理、容器 hostname 处理和原始运行顺序见 `RUN_LOG.md`；其末尾待办是采集
时刻的历史状态，以本目录说明和当前分支为准。正式 32B JSON 保留完整 argv、环境、逐次样本、
逐 rank HBM、Git provenance 和权重哈希；smoke 与 collective JSON 保留各自适用的环境、
拓扑和逐 rank/逐消息数据。
