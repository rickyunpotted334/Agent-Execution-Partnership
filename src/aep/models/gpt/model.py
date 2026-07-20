"""
GPT model ported from autoresearch-master/train.py.

Architecture:
  - Token embeddings + RMSNorm on input
  - Per-layer: scaled residual (resid_lambdas) + x0 skip (x0_lambdas)
  - CausalSelfAttention: RoPE, GQA-ready (n_kv_head), optional value embeddings (ResFormer)
  - Sliding window attention pattern ("SSSL")
  - Flash Attention 3 used when available; falls back to F.scaled_dot_product_attention
  - Squared-ReLU MLP
  - Logit soft-capping (tanh * 15)

Requires: torch  (optional dep — ImportError raised at import if absent)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

if not _TORCH_AVAILABLE:
    raise ImportError(
        "torch is required for aep.models.gpt. "
        "Install it with: pip install torch"
    )  # pragma: no cover

from aep.models.gpt.config import GPTConfig

# ---------------------------------------------------------------------------
# Optional Flash Attention 3
# ---------------------------------------------------------------------------

_FA3 = None
try:
    from kernels import get_kernel
    import torch as _t
    cap = _t.cuda.get_device_capability()
    _repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    _FA3 = get_kernel(_repo).flash_attn_interface
except Exception:
    pass  # fall back to sdpa


def _attention(q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor",
               causal: bool, window_size: tuple[int, int]) -> "torch.Tensor":
    """Dispatch to FA3 if available, otherwise use PyTorch SDPA."""
    if _FA3 is not None:
        return _FA3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    # SDPA: (B, T, H, D) → (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # Repeat k/v for GQA
    if k.size(1) != q.size(1):
        reps = q.size(1) // k.size(1)
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # (B, T, H, D)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def norm(x: "torch.Tensor") -> "torch.Tensor":
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Return True if this layer should carry a Value Embedding (alternating, last always)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(
        self,
        x: "torch.Tensor",
        ve: "torch.Tensor | None",
        cos_sin: tuple["torch.Tensor", "torch.Tensor"],
        window_size: tuple[int, int],
    ) -> "torch.Tensor":
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # type: ignore[index]
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q)
        k = norm(k)

        y = _attention(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: "torch.Tensor",
        ve: "torch.Tensor | None",
        cos_sin: tuple["torch.Tensor", "torch.Tensor"],
        window_size: tuple[int, int],
    ) -> "torch.Tensor":
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT language model from autoresearch-master, integrated into AEP.

    Architecture highlights:
      - RoPE + QK-norm per attention head
      - Value embeddings (ResFormer) on alternating layers
      - Sliding window attention with configurable pattern
      - Per-layer residual scale (resid_lambdas) + x0 skip (x0_lambdas)
      - Logit soft-capping via tanh
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin  # type: ignore[assignment]
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 10000, device: "torch.device | None" = None
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig) -> list[tuple[int, int]]:
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        char_to_window = {
            "L": (config.sequence_len, 0),
            "S": (config.sequence_len // 2, 0),
        }
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (config.sequence_len, 0)  # last layer always full context
        return sizes

    def estimate_flops(self) -> int:
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + sum(ve.weight.numel() for ve in self.value_embeds.values())
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = sum(
            12 * h * q * (t if w[0] < 0 else min(w[0], t))
            for w in self.window_sizes
        )
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self) -> dict[str, int]:
        return {
            "wte": sum(p.numel() for p in self.transformer.wte.parameters()),
            "value_embeds": sum(p.numel() for p in self.value_embeds.parameters()),
            "lm_head": sum(p.numel() for p in self.lm_head.parameters()),
            "transformer_matrices": sum(p.numel() for p in self.transformer.h.parameters()),
            "scalars": self.resid_lambdas.numel() + self.x0_lambdas.numel(),
            "total": sum(p.numel() for p in self.parameters()),
        }

    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.8, 0.95),
        scalar_lr: float = 0.5,
    ) -> "MuonAdamW":
        from aep.models.gpt.optimizer import MuonAdamW

        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        matrix_params = list(self.transformer.h.parameters())
        param_groups = [
            dict(kind="adamw", params=list(self.lm_head.parameters()),
                 lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=list(self.transformer.wte.parameters()),
                 lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=list(self.value_embeds.parameters()),
                 lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=[self.resid_lambdas],
                 lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=[self.x0_lambdas],
                 lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            param_groups.append(dict(
                kind="muon",
                params=[p for p in matrix_params if p.shape == shape],
                lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95,
                weight_decay=weight_decay,
            ))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def forward(
        self,
        idx: "torch.Tensor",
        targets: "torch.Tensor | None" = None,
        reduction: str = "mean",
    ) -> "torch.Tensor":
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]  # type: ignore[index]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15.0
        logits = softcap * torch.tanh(self.lm_head(x).float() / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
        return logits


# Re-export for convenience
__all__ = ["GPT", "GPTConfig", "Block", "CausalSelfAttention", "MLP", "has_ve", "norm"]
