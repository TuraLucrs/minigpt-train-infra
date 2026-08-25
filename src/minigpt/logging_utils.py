"""训练日志辅助函数。

本项目不引入 TensorBoard / WandB，是为了保持依赖最少。
训练日志会同时：
1. 打印到控制台，方便你边跑边看；
2. 写入 CSV，方便后续画 loss curve 或做实验对比。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


class CSVLogger:
    """一个极简 CSV logger。"""

    def __init__(self, path: str | Path, fieldnames: Iterable[str], append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

        file_exists = append and self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if append else "w"
        self.file = self.path.open(mode, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log(self, row: Dict[str, object]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
