"""A tiny character-level tokenizer implemented from scratch.

真实大模型训练一般不会用字符级 tokenizer，而会用 BPE、SentencePiece、
Unigram 等子词算法。这里故意选择 char-level，有三个原因：

1. 它足够简单，你可以完整理解 tokenizer 如何把文本变成 token ids。
2. 它没有第三方依赖，不需要 HuggingFace tokenizer。
3. 对一个训练系统学习项目来说，重点是训练链路，不是 tokenizer 算法竞赛。

注意：这个 tokenizer 只适合学习和小实验，不适合严肃训练大模型。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class CharTokenizer:
    """最小字符级 tokenizer。

    stoi: string-to-id，字符 -> token id
    itos: id-to-string，token id -> 字符

    我们保留一个 <unk> token，用来表示训练语料里没出现过的新字符。
    如果没有 <unk>，遇到未知字符时 encode 只能直接报错。
    """

    stoi: Dict[str, int]
    itos: List[str]
    unk_token: str = "<unk>"

    @classmethod
    def train_from_text(cls, text: str, extra_tokens: Iterable[str] | None = None) -> "CharTokenizer":
        """从原始文本构建词表。

        这里的“训练 tokenizer”非常朴素：统计语料里出现过哪些字符，然后排序。
        排序不是算法需要，而是为了让同一份语料每次得到稳定的 vocab。
        """

        special_tokens = list(extra_tokens or ["<unk>"])
        unique_chars = sorted(set(text))

        # special tokens 放在词表最前面，常见约定是让它们的 id 更小。
        vocab = special_tokens + [ch for ch in unique_chars if ch not in special_tokens]
        stoi = {ch: idx for idx, ch in enumerate(vocab)}
        return cls(stoi=stoi, itos=vocab, unk_token=special_tokens[0])

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        """把字符串转成 token id 列表。

        这里返回 Python list，调用方可以再转成 torch.Tensor。
        未知字符会被映射成 <unk> 的 id。
        """

        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        """把 token id 还原成字符串。

        如果 id 越界，说明输入不是这个 tokenizer 产生的，我们直接报错。
        """

        chars = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.itos):
                raise ValueError(f"Token id out of vocabulary range: {token_id}")
            chars.append(self.itos[int(token_id)])
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """保存 tokenizer 到 JSON。

        tokenizer 必须和 checkpoint 一起保存，否则模型输出的 token id 将不知道
        怎么 decode 回文本。
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "char",
            "unk_token": self.unk_token,
            "itos": self.itos,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("type") != "char":
            raise ValueError(f"Unsupported tokenizer type: {payload.get('type')}")
        itos = list(payload["itos"])
        stoi = {ch: idx for idx, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, unk_token=payload["unk_token"])
