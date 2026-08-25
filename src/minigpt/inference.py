"""MiniGPT 单设备推理基线。

v0.3 先把训练后附带的 ``model.generate()`` 拆成独立推理链路：

``文本 → tokenizer → Prefill → 第一个 token → Decode 循环 → 文本``

本版本还没有 KV Cache。Prefill 和 Decode 已经具有不同的系统语义，但 Decode 每一步仍会
把当前有效上下文重新送入完整模型。这条低效但简单的路径是 v0.4 验证 KV Cache 的 reference。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .model import MiniGPT, MiniGPTConfig, migrate_model_state_dict
from .runtime import RuntimeContext
from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    """一次生成请求真正影响 token 选择的配置。"""

    max_new_tokens: int = 32
    strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    seed: int = 1337

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.strategy not in {"greedy", "sample"}:
            raise ValueError("strategy 必须是 greedy 或 sample")
        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k 必须大于 0，或者设为 None")


@dataclass(frozen=True)
class GenerationResult:
    """一次生成的 token 结果；性能指标由 benchmark 模块单独记录。"""

    prompt_text: str
    completion_text: str
    full_text: str
    prompt_ids: list[int]
    generated_ids: list[int]
    all_ids: list[int]
    prefill_tokens: int
    stop_reason: str


class MiniGPTModelRunner:
    """把 MiniGPT 的模型 forward 暴露为 Prefill/Decode 两个推理阶段。

    v0.3 的两个方法暂时调用同一个“完整上下文 forward”。保留两个清晰入口不是为了假装
    已经优化，而是固定系统边界：v0.4 只需要改变 Decode 的内部数据流，调用者和指标定义
    不必跟着重写。
    """

    def __init__(self, model: MiniGPT, runtime: RuntimeContext) -> None:
        self.model = model
        self.runtime = runtime
        self.model.eval()

    @property
    def block_size(self) -> int:
        return self.model.config.block_size

    def _effective_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [B,T]")
        if input_ids.shape[1] == 0:
            raise ValueError("输入至少需要一个 token")
        return input_ids[:, -self.block_size :]

    def _next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        context = self._effective_context(input_ids)
        with self.runtime.autocast():
            logits = self.model(context)
        # [B,T,V] 只保留最后位置，因为它负责预测下一个 token。
        return logits[:, -1, :]

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """处理已有 prompt，返回第一个待生成 token 的 logits：[B,V]。"""

        return self._next_token_logits(input_ids)

    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """处理已经追加过 token 的上下文，返回下一个 token 的 logits：[B,V]。

        注意：v0.3 尚未缓存历史 K/V，因此这里仍重算最多 ``block_size`` 个 token。
        """

        return self._next_token_logits(input_ids)


class InferenceEngine:
    """负责 tokenizer、token 选择和 Prefill/Decode 编排的单请求推理引擎。"""

    def __init__(self, runner: MiniGPTModelRunner, tokenizer: CharTokenizer) -> None:
        self.runner = runner
        self.tokenizer = tokenizer

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt 不能为空")
        return torch.tensor([prompt_ids], dtype=torch.long, device=self.runner.runtime.device)

    def make_generator(self, config: GenerationConfig) -> torch.Generator | None:
        if config.strategy == "greedy":
            return None
        generator = torch.Generator(device=self.runner.runtime.device)
        generator.manual_seed(config.seed)
        return generator

    @staticmethod
    def select_next_token(
        next_logits: torch.Tensor,
        config: GenerationConfig,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """把每个词表 token 的分数变成被选中的下一个 token id：[B,1]。"""

        if next_logits.ndim != 2:
            raise ValueError("next_logits 必须是 [B,V]")
        if config.strategy == "greedy":
            return torch.argmax(next_logits, dim=-1, keepdim=True)

        scaled_logits = next_logits.float() / config.temperature
        if config.top_k is not None:
            k = min(config.top_k, scaled_logits.shape[-1])
            top_values, top_indices = torch.topk(scaled_logits, k=k, dim=-1)
            probabilities = torch.softmax(top_values, dim=-1)
            sampled_in_top_k = torch.multinomial(probabilities, num_samples=1, generator=generator)
            return torch.gather(top_indices, dim=-1, index=sampled_in_top_k)

        probabilities = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1, generator=generator)

    @torch.inference_mode()
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """执行一次不采集性能指标的正常生成。"""

        config.validate()
        input_ids = self.encode_prompt(prompt)
        prompt_length = input_ids.shape[1]
        generator = self.make_generator(config)

        next_logits = self.runner.prefill(input_ids)
        next_id = self.select_next_token(next_logits, config, generator)
        input_ids = torch.cat((input_ids, next_id), dim=1)

        for _ in range(1, config.max_new_tokens):
            next_logits = self.runner.decode(input_ids)
            next_id = self.select_next_token(next_logits, config, generator)
            input_ids = torch.cat((input_ids, next_id), dim=1)

        all_ids = input_ids[0].tolist()
        prompt_ids = all_ids[:prompt_length]
        generated_ids = all_ids[prompt_length:]
        completion_text = self.tokenizer.decode(generated_ids)
        return GenerationResult(
            prompt_text=prompt,
            completion_text=completion_text,
            full_text=self.tokenizer.decode(all_ids),
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            all_ids=all_ids,
            prefill_tokens=min(prompt_length, self.runner.block_size),
            stop_reason="length",
        )


def checkpoint_run_dir(checkpoint_path: Path) -> Path:
    """返回 checkpoint 所属 run 目录。"""

    return checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent


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
) -> InferenceEngine:
    """从 v0.2.x 训练产物恢复只用于推理的模型和 tokenizer。"""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    tokenizer = CharTokenizer.load(_find_tokenizer_path(checkpoint_path, checkpoint, tokenizer_path))

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
    model.load_state_dict(migrate_model_state_dict(checkpoint["model_state"]), strict=True)
    model.to(runtime.device)
    model.eval()
    return InferenceEngine(MiniGPTModelRunner(model, runtime), tokenizer)
