"""Qwen3 tokenizer、runner 和通用 InferenceEngine 装配。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from .inference import InferenceEngine, _validate_prefill_inputs
from .qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3KVCache, load_qwen3_from_pretrained
from .runtime import RuntimeContext


def _load_generation_eos_token_ids(
    model_dir: str | Path,
    fallback_eos_token_id: int | None,
) -> tuple[int, ...]:
    """读取 HF generation_config.json；缺失时退回 tokenizer 的单 EOS。"""

    path = Path(model_dir) / "generation_config.json"
    value: object = fallback_eos_token_id
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取 Qwen3 generation config: {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Qwen3 generation_config.json 顶层必须是对象")
        value = raw.get("eos_token_id", fallback_eos_token_id)

    if value is None:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        token_ids = (value,)
    elif isinstance(value, list) and all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in value
    ):
        token_ids = tuple(value)
    else:
        raise ValueError("generation_config.json 的 eos_token_id 必须是整数或整数列表")
    if not token_ids or any(token_id < 0 for token_id in token_ids):
        raise ValueError("generation_config.json 的 eos_token_id 不能为空或包含负数")
    return tuple(dict.fromkeys(token_ids))


class Qwen3Tokenizer:
    """限制为本地文件的 Transformers tokenizer 适配器。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        use_chat_template: bool = False,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen3 tokenizer 需要安装 transformers") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise ValueError("Qwen3 tokenizer 同时缺少 pad_token_id 和 eos_token_id")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._generation_eos_token_ids = _load_generation_eos_token_ids(
            model_dir,
            self.eos_token_id,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def pad_token_id(self) -> int:
        value = self._tokenizer.pad_token_id
        if value is None:
            raise RuntimeError("Qwen3 tokenizer pad_token_id 未初始化")
        return int(value)

    @property
    def eos_token_id(self) -> int | None:
        value = self._tokenizer.eos_token_id
        return None if value is None else int(value)

    @property
    def generation_eos_token_ids(self) -> tuple[int, ...]:
        return self._generation_eos_token_ids

    def encode(self, text: str) -> list[int]:
        if self.use_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": text})
            templated = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            # transformers 4.x 返回 list[int]；5.x 返回 BatchEncoding，
            # 其 input_ids 可能是 list[int] 或带 batch 维的张量。
            if hasattr(templated, "keys") and "input_ids" in templated:
                templated = templated["input_ids"]
            if hasattr(templated, "tolist"):
                templated = templated.tolist()
            if templated and isinstance(templated[0], (list, tuple)):
                if len(templated) != 1:
                    raise ValueError("chat template 必须返回单行 token 序列")
                templated = templated[0]
            return [int(token_id) for token_id in templated]
        return [
            int(token_id)
            for token_id in self._tokenizer.encode(text, add_special_tokens=False)
        ]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )


def _last_valid_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """返回每行最后一个有效 token 的物理列号，兼容左/右 padding。"""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask 必须是 [B,T]")
    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).expand_as(attention_mask)
    masked_positions = positions.masked_fill(~attention_mask.bool(), -1)
    last_positions = masked_positions.max(dim=1).values
    if torch.any(last_positions < 0):
        raise ValueError("每个 prompt 至少需要一个有效 token")
    return last_positions


def _right_padded_histories(
    histories: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(int(row.numel()) for row in histories)
    input_ids = torch.zeros((len(histories), width), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_index, row in enumerate(histories):
        length = int(row.numel())
        input_ids[row_index, :length] = row
        attention_mask[row_index, :length] = True
    return input_ids, attention_mask


class RecomputeQwen3ModelRunner:
    implementation_name = "qwen3_recompute"

    def __init__(self, model: Qwen3ForCausalLM, runtime: RuntimeContext) -> None:
        self.model = model.eval()
        self.runtime = runtime
        self._histories: list[torch.Tensor] = []

    @property
    def block_size(self) -> int:
        return self.model.config.max_position_embeddings

    def validate_generation(
        self,
        prompt_lengths: Sequence[int],
        max_new_tokens: int,
    ) -> None:
        required = max(prompt_lengths) + max_new_tokens - 1
        if required > self.block_size:
            raise ValueError(
                f"prompt + max_new_tokens 需要 {required} 个位置，超过 Qwen3 上限 "
                f"{self.block_size}"
            )

    def _forward_histories(self) -> torch.Tensor:
        input_ids, attention_mask = _right_padded_histories(
            self._histories,
            device=self.runtime.device,
        )
        last_positions = _last_valid_positions(attention_mask)
        with self.runtime.autocast():
            return self.model(
                input_ids,
                attention_mask,
                logit_positions=last_positions,
            )

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_prefill_inputs(input_ids, attention_mask)
        self._histories = [
            input_ids[row][attention_mask[row].bool()].clone()
            for row in range(input_ids.shape[0])
        ]
        return self._forward_histories()

    def decode(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(self._histories)
        if input_ids.shape != (batch_size, 1) or active_mask.shape != (batch_size,):
            raise ValueError("Qwen3 Decode 需要 [B,1] input_ids 和 [B] active_mask")
        for row in range(batch_size):
            if bool(active_mask[row].item()):
                self._histories[row] = torch.cat((self._histories[row], input_ids[row]))
        return self._forward_histories()


class CachedQwen3ModelRunner:
    implementation_name = "qwen3_kv_cache"

    def __init__(self, model: Qwen3ForCausalLM, runtime: RuntimeContext) -> None:
        self.model = model.eval()
        self.runtime = runtime
        self._cache: Qwen3KVCache | None = None
        self._planned_max_seq_len: int | None = None

    @property
    def block_size(self) -> int:
        return self.model.config.max_position_embeddings

    @property
    def cache(self) -> Qwen3KVCache | None:
        return self._cache

    def release_cache(self) -> None:
        self._cache = None
        self._planned_max_seq_len = None

    def validate_generation(
        self,
        prompt_lengths: Sequence[int],
        max_new_tokens: int,
    ) -> None:
        # 第一个输出 token 直接来自 Prefill logits；只有后续 token 才需要把前一个
        # 输出写入 KV Cache，所以 N 个新 token 只新增 N-1 个模型输入位置。
        required = max(prompt_lengths) + max_new_tokens - 1
        if required > self.block_size:
            raise ValueError(
                f"prompt + max_new_tokens 需要 {required} 个位置，超过 Qwen3 上限 "
                f"{self.block_size}"
            )
        self._planned_max_seq_len = required

    def _ensure_cache(self, batch_size: int, prompt_length: int) -> Qwen3KVCache:
        capacity = self._planned_max_seq_len or min(self.block_size, prompt_length + 1)
        dtype = next(self.model.parameters()).dtype
        needs_allocation = (
            self._cache is None
            or self._cache.max_batch_size < batch_size
            or self._cache.max_seq_len < capacity
            or self._cache.layers[0].key.dtype != dtype
        )
        if needs_allocation:
            self._cache = self.model.allocate_kv_cache(
                batch_size,
                capacity,
                device=self.runtime.device,
                dtype=dtype,
            )
        return self._cache

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_prefill_inputs(input_ids, attention_mask)
        last_positions = _last_valid_positions(attention_mask)
        max_prompt_length = int(attention_mask.long().sum(dim=1).max().item())
        cache = self._ensure_cache(input_ids.shape[0], max_prompt_length)
        with self.runtime.autocast():
            return self.model.prefill_with_cache(
                input_ids,
                attention_mask,
                cache,
                logit_positions=last_positions,
            )

    def decode(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("Qwen3 Decode 前必须先执行 Prefill")
        with self.runtime.autocast():
            logits = self.model.decode_with_cache(
                input_ids,
                active_mask.bool(),
                self._cache,
            )
        return logits[:, 0]


def validate_qwen3_tokenizer_config(
    tokenizer: Qwen3Tokenizer,
    config: Qwen3Config,
) -> None:
    """确保 tokenizer、generation config 与模型输出词表使用同一编号空间。"""

    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError(
            "tokenizer 词表不能大于模型输出词表："
            f"tokenizer={tokenizer.vocab_size}, model={config.vocab_size}"
        )
    if tokenizer.pad_token_id >= config.vocab_size:
        raise ValueError("tokenizer pad_token_id 超出模型词表")
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= config.vocab_size:
        raise ValueError("tokenizer eos_token_id 超出模型词表")
    if any(
        token_id >= config.vocab_size
        for token_id in tokenizer.generation_eos_token_ids
    ):
        raise ValueError("generation config 停止 token 超出模型词表")


def load_qwen3_engine(
    model_dir: str | Path,
    runtime: RuntimeContext,
    *,
    decode_mode: str = "kv_cache",
    use_chat_template: bool = False,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> InferenceEngine:
    """从本地 Hugging Face 目录加载 tokenizer、权重和所选 runner。"""

    dtype = runtime.amp_dtype or torch.float32
    tokenizer = Qwen3Tokenizer(
        model_dir,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    model = load_qwen3_from_pretrained(
        model_dir,
        device=runtime.device,
        dtype=dtype,
    )
    validate_qwen3_tokenizer_config(tokenizer, model.config)
    if decode_mode == "kv_cache":
        runner = CachedQwen3ModelRunner(model, runtime)
    elif decode_mode == "recompute":
        runner = RecomputeQwen3ModelRunner(model, runtime)
    else:
        raise ValueError("decode_mode 必须是 kv_cache 或 recompute")
    return InferenceEngine(
        runner,
        tokenizer,
        pad_token_id=tokenizer.pad_token_id,
    )
