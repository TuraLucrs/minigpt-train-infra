"""MiniGPT 的 Prefill/Decode、KV Cache 与静态 batch 推理编排。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch

from .checkpoint import load_checkpoint
from .model import MiniGPT, MiniGPTConfig, MiniGPTKVCache, migrate_model_state_dict
from .runtime import RuntimeContext
from .tokenizer import CharTokenizer


class TextTokenizer(Protocol):
    """推理引擎实际依赖的最小 tokenizer 接口。"""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class GenerationConfig:
    """一次生成请求真正影响 token 选择和停止条件的配置。"""

    max_new_tokens: int = 32
    strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 1337
    eos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] | None = None

    def stop_token_ids(self) -> tuple[int, ...]:
        """返回去重后的停止 token；保留 singular 字段以兼容旧调用方。"""

        values = self.eos_token_ids
        if values is None:
            values = () if self.eos_token_id is None else (self.eos_token_id,)
        return tuple(dict.fromkeys(values))

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.strategy not in {"greedy", "sample"}:
            raise ValueError("strategy 必须是 greedy 或 sample")
        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k 必须大于 0，或者设为 None")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p 必须位于 (0, 1]，或者设为 None")
        if self.eos_token_id is not None and self.eos_token_ids is not None:
            raise ValueError("eos_token_id 和 eos_token_ids 不能同时设置")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id 不能小于 0")
        if self.eos_token_ids is not None:
            if not self.eos_token_ids:
                raise ValueError("eos_token_ids 不能为空；不停止时请设为 None")
            if any(token_id < 0 for token_id in self.eos_token_ids):
                raise ValueError("eos_token_ids 不能包含负数")


@dataclass(frozen=True)
class GenerationResult:
    """一次生成的 token 与文本结果；性能指标由 benchmark 模块记录。"""

    prompt_text: str
    completion_text: str
    full_text: str
    prompt_ids: list[int]
    generated_ids: list[int]
    all_ids: list[int]
    prefill_tokens: int
    stop_reason: str


class ModelRunner(Protocol):
    """不同模型和 Decode 实现交给通用引擎的最小接口。"""

    runtime: RuntimeContext
    implementation_name: str

    @property
    def block_size(self) -> int: ...

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def decode(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor: ...


def _validate_prefill_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids/attention_mask 必须同为 [B,T]")
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("Prefill 至少需要一个 batch 和一个 token")
    if torch.any(attention_mask.long().sum(dim=1) <= 0):
        raise ValueError("每个 prompt 至少需要一个有效 token")


def _right_padded_rows(
    rows: Sequence[torch.Tensor],
    *,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """截取每行最后一个窗口，再组成右侧 padding 的 `[B,T]`。"""

    windows = [row[-max_length:] for row in rows]
    width = max(int(row.numel()) for row in windows)
    input_ids = torch.zeros((len(windows), width), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_index, row in enumerate(windows):
        row_length = int(row.numel())
        input_ids[row_index, :row_length] = row
        attention_mask[row_index, :row_length] = True
    return input_ids, attention_mask


class RecomputeMiniGPTModelRunner:
    """每个 Decode step 重算有效上下文的正确性参考实现。"""

    implementation_name = "recompute"

    def __init__(self, model: MiniGPT, runtime: RuntimeContext) -> None:
        self.model = model.eval()
        self.runtime = runtime
        self._histories: list[torch.Tensor] = []

    @property
    def block_size(self) -> int:
        return self.model.config.block_size

    def _forward_histories(self) -> torch.Tensor:
        input_ids, attention_mask = _right_padded_rows(
            self._histories,
            device=self.runtime.device,
            max_length=self.block_size,
        )
        with self.runtime.autocast():
            logits = self.model(input_ids, attention_mask)
        last_positions = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        return logits[batch_indices, last_positions]

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
            raise ValueError("Decode 需要 [B,1] input_ids 和 [B] active_mask")
        for row in range(batch_size):
            if bool(active_mask[row].item()):
                self._histories[row] = torch.cat((self._histories[row], input_ids[row]))
        return self._forward_histories()


class CachedMiniGPTModelRunner:
    """使用逐层预分配 KV Cache 的 MiniGPT Decode 实现。"""

    implementation_name = "kv_cache"

    def __init__(self, model: MiniGPT, runtime: RuntimeContext) -> None:
        self.model = model.eval()
        self.runtime = runtime
        self._cache: MiniGPTKVCache | None = None
        self._window_ids: torch.Tensor | None = None
        self._window_lengths: list[int] | None = None

    @property
    def block_size(self) -> int:
        return self.model.config.block_size

    @property
    def cache(self) -> MiniGPTKVCache | None:
        return self._cache

    def release_cache(self) -> None:
        """释放缓存引用；设备 allocator 是否归还内存由后端决定。"""

        self._cache = None
        self._window_ids = None
        self._window_lengths = None

    def _ensure_cache(self, batch_size: int) -> MiniGPTKVCache:
        dtype = self.runtime.amp_dtype or next(self.model.parameters()).dtype
        needs_allocation = (
            self._cache is None
            or self._cache.max_batch_size < batch_size
            or self._cache.layers[0].key.dtype != dtype
        )
        if needs_allocation:
            self._cache = self.model.allocate_kv_cache(
                batch_size,
                device=self.runtime.device,
                dtype=dtype,
            )
        return self._cache

    def _prefill_window(self) -> torch.Tensor:
        if self._window_ids is None or self._window_lengths is None:
            raise RuntimeError("缓存窗口尚未初始化")
        rows = [
            self._window_ids[row, : self._window_lengths[row]]
            for row in range(self._window_ids.shape[0])
        ]
        input_ids, attention_mask = _right_padded_rows(
            rows,
            device=self.runtime.device,
            max_length=self.block_size,
        )
        with self.runtime.autocast():
            logits = self.model.prefill_with_cache(
                input_ids,
                attention_mask,
                self._ensure_cache(input_ids.shape[0]),
            )
        last_positions = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        return logits[batch_indices, last_positions]

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_prefill_inputs(input_ids, attention_mask)
        rows = [
            input_ids[row][attention_mask[row].bool()][-self.block_size :].clone()
            for row in range(input_ids.shape[0])
        ]
        self._window_ids = torch.zeros(
            (len(rows), self.block_size),
            dtype=torch.long,
            device=self.runtime.device,
        )
        self._window_lengths = [int(row.numel()) for row in rows]
        for row_index, row in enumerate(rows):
            self._window_ids[row_index, : row.numel()] = row
        return self._prefill_window()

    def decode(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._cache is None or self._window_ids is None or self._window_lengths is None:
            raise RuntimeError("Decode 前必须先执行 Prefill")
        batch_size = self._window_ids.shape[0]
        if input_ids.shape != (batch_size, 1) or active_mask.shape != (batch_size,):
            raise ValueError("Decode 需要 [B,1] input_ids 和 [B] active_mask")

        active_rows = [bool(value) for value in active_mask.tolist()]
        requires_window_rebuild = any(
            active_rows[row] and self._window_lengths[row] >= self.block_size
            for row in range(batch_size)
        )
        if requires_window_rebuild:
            # MiniGPT 使用 learned absolute position。窗口左移后所有保留 token 的位置都会改变，
            # 因而必须重建 K/V；Qwen3 的 RoPE 路径不会沿用这个限制。
            for row in range(batch_size):
                if not active_rows[row]:
                    continue
                length = self._window_lengths[row]
                if length >= self.block_size:
                    self._window_ids[row, :-1] = self._window_ids[row, 1:].clone()
                    self._window_ids[row, -1] = input_ids[row, 0]
                else:
                    self._window_ids[row, length] = input_ids[row, 0]
                    self._window_lengths[row] += 1
            return self._prefill_window()

        for row in range(batch_size):
            if active_rows[row]:
                position = self._window_lengths[row]
                self._window_ids[row, position] = input_ids[row, 0]
                self._window_lengths[row] += 1
        with self.runtime.autocast():
            logits = self.model.decode_with_cache(
                input_ids,
                active_mask.bool(),
                self._cache,
            )
        return logits[:, 0]


class SlotCachedMiniGPTModelRunner:
    """为 Continuous Batching 预分配稳定 cache slots 的 MiniGPT runner。"""

    implementation_name = "minigpt_slot_kv_cache"

    def __init__(
        self,
        model: MiniGPT,
        runtime: RuntimeContext,
        *,
        max_slots: int,
    ) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots 必须大于 0")
        self.model = model.eval()
        self.runtime = runtime
        self.max_slots = max_slots
        self.max_seq_len = model.config.block_size
        dtype = runtime.amp_dtype or next(model.parameters()).dtype
        self._cache = model.allocate_kv_cache(
            max_slots,
            device=runtime.device,
            dtype=dtype,
        )

    @property
    def block_size(self) -> int:
        return self.model.config.block_size

    @property
    def cache(self) -> MiniGPTKVCache:
        return self._cache

    def _rows(self, slot_ids: Sequence[int]) -> torch.Tensor:
        if not slot_ids:
            raise ValueError("一次 slot 模型调用至少需要一个请求")
        return torch.tensor(slot_ids, dtype=torch.long, device=self.runtime.device)

    def validate_request(self, prompt_length: int, max_new_tokens: int) -> None:
        required = prompt_length + max_new_tokens - 1
        if prompt_length <= 0:
            raise ValueError("prompt 至少需要一个 token")
        if required > self.max_seq_len:
            raise ValueError(
                f"prompt + max_new_tokens 需要 {required} 个 cache 位置，"
                f"超过 MiniGPT 上限 {self.max_seq_len}"
            )

    def prefill_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_prefill_inputs(input_ids, attention_mask)
        if input_ids.shape[0] != len(slot_ids):
            raise ValueError("slot_ids 必须与 Prefill batch 一一对应")
        rows = self._rows(slot_ids)
        with self.runtime.autocast():
            logits = self.model.prefill_with_cache(
                input_ids,
                attention_mask,
                self._cache,
                cache_rows=rows,
            )
        last_positions = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        return logits[batch_indices, last_positions]

    def decode_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.shape != (len(slot_ids), 1):
            raise ValueError("slot Decode 需要与 slot_ids 等长的 [B,1] input_ids")
        rows = self._rows(slot_ids)
        active = torch.ones(len(slot_ids), dtype=torch.bool, device=input_ids.device)
        with self.runtime.autocast():
            return self.model.decode_with_cache(
                input_ids,
                active,
                self._cache,
                cache_rows=rows,
            )[:, 0]

    def release_slots(self, slot_ids: Sequence[int]) -> None:
        if not slot_ids:
            return
        rows = self._rows(slot_ids)
        self._cache.lengths[rows].zero_()
        self._cache.current_max_length = int(self._cache.lengths.max().item())

    def cache_lengths(self, slot_ids: Sequence[int]) -> list[int]:
        if not slot_ids:
            return []
        return [int(value) for value in self._cache.lengths[self._rows(slot_ids)].tolist()]


# 保留 v0.3 公共名称，旧调用方默认得到 recompute reference runner。
MiniGPTModelRunner = RecomputeMiniGPTModelRunner


class InferenceEngine:
    """负责 tokenizer、token 选择和 Prefill/Decode 编排。"""

    def __init__(
        self,
        runner: ModelRunner,
        tokenizer: TextTokenizer,
        *,
        pad_token_id: int | None = None,
    ) -> None:
        self.runner = runner
        self.tokenizer = tokenizer
        if pad_token_id is None:
            try:
                pad_token_id = tokenizer.stoi[tokenizer.unk_token]  # type: ignore[attr-defined]
            except AttributeError as exc:
                raise ValueError("非 CharTokenizer 必须显式提供 pad_token_id") from exc
        self.pad_token_id = int(pad_token_id)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt 不能为空")
        return torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.runner.runtime.device,
        )

    def encode_prompts(
        self,
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        if not prompts:
            raise ValueError("prompts 不能为空")
        prompt_rows = [self.tokenizer.encode(prompt) for prompt in prompts]
        if any(not row for row in prompt_rows):
            raise ValueError("每个 prompt 都必须至少包含一个 token")

        width = max(len(row) for row in prompt_rows)
        input_ids = torch.full(
            (len(prompt_rows), width),
            self.pad_token_id,
            dtype=torch.long,
            device=self.runner.runtime.device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_index, row in enumerate(prompt_rows):
            row_length = len(row)
            input_ids[row_index, :row_length] = torch.tensor(row, device=input_ids.device)
            attention_mask[row_index, :row_length] = True
        return input_ids, attention_mask, prompt_rows

    def validate_generation_capacity(
        self,
        prompt_rows: Sequence[Sequence[int]],
        max_new_tokens: int,
    ) -> None:
        validate = getattr(self.runner, "validate_generation", None)
        if callable(validate):
            validate([len(row) for row in prompt_rows], max_new_tokens)

    def make_generator(self, config: GenerationConfig) -> torch.Generator | None:
        if config.strategy == "greedy":
            return None
        generator = torch.Generator(device=self.runner.runtime.device)
        generator.manual_seed(config.seed)
        return generator

    def make_generators(
        self,
        config: GenerationConfig,
        batch_size: int,
    ) -> list[torch.Generator | None]:
        if config.strategy == "greedy":
            return [None] * batch_size
        generators: list[torch.Generator | None] = []
        for row in range(batch_size):
            generator = torch.Generator(device=self.runner.runtime.device)
            generator.manual_seed(config.seed + row)
            generators.append(generator)
        return generators

    @staticmethod
    def select_next_token(
        next_logits: torch.Tensor,
        config: GenerationConfig,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if next_logits.ndim != 2:
            raise ValueError("next_logits 必须是 [B,V]")
        if config.strategy == "greedy":
            return torch.argmax(next_logits, dim=-1, keepdim=True)

        scaled_logits = next_logits.float() / config.temperature
        candidate_indices = torch.arange(
            scaled_logits.shape[-1],
            device=scaled_logits.device,
        ).expand_as(scaled_logits)
        if config.top_k is not None:
            k = min(config.top_k, scaled_logits.shape[-1])
            scaled_logits, candidate_indices = torch.topk(
                scaled_logits,
                k=k,
                dim=-1,
            )
        if config.top_p is not None and config.top_p < 1.0:
            scaled_logits, order = torch.sort(scaled_logits, descending=True, dim=-1)
            candidate_indices = torch.gather(candidate_indices, dim=-1, index=order)
            cumulative = torch.softmax(scaled_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative > config.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            scaled_logits = scaled_logits.masked_fill(remove, float("-inf"))
        sampled_index = torch.multinomial(
            torch.softmax(scaled_logits, dim=-1),
            num_samples=1,
            generator=generator,
        )
        return torch.gather(candidate_indices, dim=-1, index=sampled_index)

    def synchronize_next_ids(self, next_ids: torch.Tensor) -> torch.Tensor:
        """允许分布式引擎在进入下一次 Decode 前统一 token；单进程原样返回。"""

        return next_ids

    @torch.inference_mode()
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        return self.generate_batch([prompt], config)[0]

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        config.validate()
        stop_token_ids = frozenset(config.stop_token_ids())
        if any(token_id >= self.tokenizer.vocab_size for token_id in stop_token_ids):
            raise ValueError("停止 token 超出 tokenizer 词表")

        input_ids, attention_mask, prompt_rows = self.encode_prompts(prompts)
        self.validate_generation_capacity(prompt_rows, config.max_new_tokens)
        batch_size = len(prompts)
        generators = self.make_generators(config, batch_size)
        generated_ids: list[list[int]] = [[] for _ in prompts]
        stop_reasons = ["length"] * batch_size
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        next_logits = self.runner.prefill(input_ids, attention_mask)
        for step in range(config.max_new_tokens):
            next_ids = torch.cat(
                [
                    self.select_next_token(
                        next_logits[row : row + 1],
                        config,
                        generators[row],
                    )
                    for row in range(batch_size)
                ]
            )
            next_ids = self.synchronize_next_ids(next_ids)
            for row in range(batch_size):
                if bool(finished[row].item()):
                    next_ids[row, 0] = self.pad_token_id
                    continue
                token_id = int(next_ids[row, 0].item())
                generated_ids[row].append(token_id)
                if token_id in stop_token_ids:
                    finished[row] = True
                    stop_reasons[row] = "eos"

            if step + 1 == config.max_new_tokens or bool(torch.all(finished).item()):
                break
            next_logits = self.runner.decode(next_ids, ~finished)

        results = []
        for row, prompt in enumerate(prompts):
            all_ids = prompt_rows[row] + generated_ids[row]
            results.append(
                GenerationResult(
                    prompt_text=prompt,
                    completion_text=self.tokenizer.decode(generated_ids[row]),
                    full_text=self.tokenizer.decode(all_ids),
                    prompt_ids=prompt_rows[row],
                    generated_ids=generated_ids[row],
                    all_ids=all_ids,
                    prefill_tokens=min(len(prompt_rows[row]), self.runner.block_size),
                    stop_reason=stop_reasons[row],
                )
            )
        return results


def checkpoint_run_dir(checkpoint_path: Path) -> Path:
    """返回 checkpoint 所属 run 目录。"""

    return (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )


def _find_tokenizer_path(
    checkpoint_path: Path,
    checkpoint: dict,
    explicit_path: str | Path | None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(checkpoint_run_dir(checkpoint_path) / "tokenizer.json")

    stored_path = checkpoint.get("tokenizer_path")
    if stored_path is None:
        stored_path = checkpoint.get("config", {}).get("runtime", {}).get("tokenizer_path")
    if stored_path:
        candidates.append(Path(stored_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"找不到与 checkpoint 对应的 tokenizer.json，已检查：\n{checked}")


def load_minigpt_engine(
    checkpoint_path: str | Path,
    runtime: RuntimeContext,
    tokenizer_path: str | Path | None = None,
    decode_mode: str = "kv_cache",
) -> InferenceEngine:
    """从 v0.2.x 训练产物恢复模型和 tokenizer，并选择 Decode 实现。"""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    tokenizer = CharTokenizer.load(
        _find_tokenizer_path(checkpoint_path, checkpoint, tokenizer_path)
    )
    raw_model_config = checkpoint.get("config", {}).get("model")
    if not isinstance(raw_model_config, dict):
        raise ValueError("checkpoint 缺少 config.model，无法重建 MiniGPT 结构")

    model_config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=int(raw_model_config["block_size"]),
        n_layer=int(raw_model_config["n_layer"]),
        n_head=int(raw_model_config["n_head"]),
        n_embd=int(raw_model_config["n_embd"]),
        dropout=float(raw_model_config.get("dropout", 0.0)),
    )
    model = MiniGPT(model_config)
    model.load_state_dict(
        migrate_model_state_dict(checkpoint["model_state"]),
        strict=True,
    )
    model.to(runtime.device).eval()

    if decode_mode == "kv_cache":
        runner: ModelRunner = CachedMiniGPTModelRunner(model, runtime)
    elif decode_mode == "recompute":
        runner = RecomputeMiniGPTModelRunner(model, runtime)
    else:
        raise ValueError("decode_mode 必须是 kv_cache 或 recompute")
    return InferenceEngine(runner, tokenizer)
