"""Small smoke tests for MiniGPT-Train.

这个文件不追求覆盖所有训练行为，只检查最核心的部件能不能协同工作：
- tokenizer 能 encode/decode
- model forward shape 正确
- 手写 loss 能 backward
- 手写 AdamW 能更新参数
- checkpoint 能保存和加载

运行：
    python tests/test_core.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.checkpoint import build_checkpoint_payload, load_checkpoint, save_checkpoint  # noqa: E402
from minigpt.data import RandomTokenBatcher  # noqa: E402
from minigpt.model import MiniGPT, MiniGPTConfig, next_token_cross_entropy  # noqa: E402
from minigpt.optim import MiniAdamW, SimpleGradScaler  # noqa: E402
from minigpt.tokenizer import CharTokenizer  # noqa: E402
from train import checkpoint_run_dir, find_resume_tokenizer_path, tokenizer_fingerprint, validate_resume_metadata  # noqa: E402


def main() -> None:
    text = "hello minigpt\nhello training\n"
    tokenizer = CharTokenizer.train_from_text(text)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text

    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = MiniGPT(config)
    optimizer = MiniAdamW(model.parameters(), lr=1e-3)
    scaler = SimpleGradScaler(enabled=False)

    x = torch.tensor([ids[:8], ids[1:9]], dtype=torch.long)
    y = torch.tensor([ids[1:9], ids[2:10]], dtype=torch.long)

    logits = model(x)
    assert logits.shape == (2, 8, tokenizer.vocab_size)

    loss = next_token_cross_entropy(logits, y)
    expected_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
    assert torch.allclose(loss, expected_loss, atol=1e-6)
    assert loss.ndim == 0

    try:
        next_token_cross_entropy(logits, torch.full_like(y, tokenizer.vocab_size), debug_checks=True)
    except ValueError:
        pass
    else:
        raise AssertionError("debug_checks should reject token ids outside the vocabulary")

    loss.backward()

    # A sequence of block_size + 1 tokens is exactly enough to form one GPT sample.
    exact_batcher = RandomTokenBatcher(
        tokens=torch.arange(9),
        batch_size=4,
        block_size=8,
        device=torch.device("cpu"),
        seed=123,
    )
    bx, by = exact_batcher.get_batch()
    assert bx.shape == (4, 8)
    assert by.shape == (4, 8)
    assert torch.equal(by, bx + 1)

    # With 10 tokens and block_size 8, starts 0 and 1 are both legal.
    # The old off-by-one bug silently made start 1 impossible.
    boundary_batcher = RandomTokenBatcher(
        tokens=torch.arange(10),
        batch_size=256,
        block_size=8,
        device=torch.device("cpu"),
        seed=123,
    )
    bx, by = boundary_batcher.get_batch()
    assert torch.equal(by, bx + 1)
    assert torch.any(bx[:, 0] == 1)

    before = model.token_embedding.weight.detach().clone()
    optimizer.step()
    optimizer.zero_grad()
    after = model.token_embedding.weight.detach().clone()
    assert not torch.equal(before, after)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=3,
            config={"test": True},
            best_val_loss=1.23,
        )
        save_checkpoint(ckpt_path, payload)
        loaded = load_checkpoint(ckpt_path, map_location="cpu")
        assert loaded["step"] == 3
        assert loaded["config"]["test"] is True
        assert "model_state" in loaded

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        tokenizer_path = run_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)

        latest_path = run_dir / "latest.pt"
        step_path = checkpoints_dir / "step_000003.pt"
        latest_path.write_bytes(b"not a real checkpoint")
        step_path.write_bytes(b"not a real checkpoint")

        checkpoint = {
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_hash": tokenizer_fingerprint(tokenizer),
            "data_hash": "abc",
            "config": {"runtime": {}},
        }
        assert checkpoint_run_dir(latest_path) == run_dir
        assert checkpoint_run_dir(step_path) == run_dir
        assert find_resume_tokenizer_path(step_path, checkpoint, Path(tmpdir) / "new_run") == tokenizer_path
        validate_resume_metadata(checkpoint, tokenizer, tokenizer_fingerprint(tokenizer), "abc")

        bad_checkpoint = dict(checkpoint)
        bad_checkpoint["tokenizer_hash"] = "definitely-wrong"
        try:
            validate_resume_metadata(bad_checkpoint, tokenizer, tokenizer_fingerprint(tokenizer), "abc")
        except ValueError:
            pass
        else:
            raise AssertionError("resume metadata validation should reject tokenizer hash mismatches")

    print("All core smoke tests passed.")


if __name__ == "__main__":
    main()
