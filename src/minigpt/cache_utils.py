"""KV Cache 槽位索引的公共校验。"""

from __future__ import annotations

import torch


def normalize_cache_rows(
    cache_rows: torch.Tensor | None,
    *,
    batch_size: int,
    max_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """返回与输入 batch 一一对应、互不重复的 cache 行号。"""

    if batch_size <= 0:
        raise ValueError("cache batch_size 必须大于 0")
    if cache_rows is None:
        if batch_size > max_batch_size:
            raise ValueError("batch_size 超出 KV Cache 槽位容量")
        return torch.arange(batch_size, dtype=torch.long, device=device)
    if cache_rows.ndim != 1 or cache_rows.shape[0] != batch_size:
        raise ValueError("cache_rows 必须是与输入 batch 等长的 [B] Tensor")
    rows = cache_rows.to(device=device, dtype=torch.long)
    if bool(torch.any(rows < 0).item()) or bool(torch.any(rows >= max_batch_size).item()):
        raise ValueError("cache_rows 包含越界槽位")
    if torch.unique(rows).numel() != rows.numel():
        raise ValueError("同一次模型调用不能重复使用同一个 cache slot")
    return rows
