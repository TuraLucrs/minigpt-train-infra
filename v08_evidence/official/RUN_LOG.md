# v0.8 Decode vocab collective A/B

- complete: `false`
- status: `incomplete`
- 结论：证据门禁未通过，不能判断词表通信优化是否值得继续。

| case | raw throughput speedup | goodput speedup | TPOT reduction | communication reduction |
|---|---:|---:|---:|---:|
| short_decode | 0.9888× | 0.9888× | 1.44% | -8.81% |
| mixed_decode | 1.0017× | 1.0094× | -1.48% | -0.89% |

## Incomplete

- short_decode/session-01-full_gather: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
- short_decode/session-02-distributed_argmax: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
- short_decode/session-03-distributed_argmax: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
- short_decode/session-04-full_gather: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
- mixed_decode/session-01-full_gather: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
- mixed_decode/session-02-distributed_argmax: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
- mixed_decode/session-03-distributed_argmax: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
- mixed_decode/session-04-full_gather: 正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'

## Warnings

- short_decode 跨 A/B 存在逐 token 数值变体
- mixed_decode 跨 A/B 存在逐 token 数值变体
