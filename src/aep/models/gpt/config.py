from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    """Architecture configuration for the AEP GPT model.

    Sourced from autoresearch-master/train.py and adapted for AEP.
    The model uses:
      - Rotary position embeddings (RoPE)
      - Value embeddings (ResFormer-style residual gating)
      - Sliding window attention (pattern: S=half-ctx, L=full-ctx)
      - RMSNorm
      - Squared-ReLU MLP
    """

    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"  # repeating pattern: L=full window, S=half window

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def small(cls) -> "GPTConfig":
        """Tiny config suitable for CPU tests (depth=2, embd=64)."""
        return cls(
            sequence_len=64,
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="SL",
        )

    @classmethod
    def from_depth(cls, depth: int, vocab_size: int, aspect_ratio: int = 64, head_dim: int = 128) -> "GPTConfig":
        """Build config from depth using aspect-ratio scaling (from train.py)."""
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim
        return cls(
            sequence_len=2048,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            window_pattern="SSSL",
        )
