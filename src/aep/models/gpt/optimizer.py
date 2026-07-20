"""
MuonAdamW optimizer ported from autoresearch-master/train.py.

Muon (Momentum + Orthogonalization Using Newton-Schulz) is used for 2-D matrix
parameters (transformer weight matrices). AdamW is used for all other parameters.

The NorMuon variant applies variance reduction before the parameter update.
Polar Express Newton-Schulz coefficients are used for fast matrix orthogonalization.

torch.compile is applied to the fused step kernels when available; falls back
to eager execution on CPU or when torch.compile is unsupported.
"""
from __future__ import annotations

from typing import Any

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

if not _TORCH_AVAILABLE:
    raise ImportError(
        "torch is required for aep.models.gpt.optimizer. "
        "Install it with: pip install torch"
    )  # pragma: no cover

# ---------------------------------------------------------------------------
# Polar Express Newton-Schulz coefficients for fast matrix orthogonalization
# ---------------------------------------------------------------------------

_POLAR_EXPRESS_COEFFS: list[tuple[float, float, float]] = [
    (8.156554524902461,  -22.48329292557795,  15.878769915207462),
    (4.042929935166739,   -2.808917465908714,  0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,  0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081, 0.42323551169305323),
]


def _try_compile(fn: Any) -> Any:
    """Apply torch.compile when available; return fn unchanged otherwise."""
    try:
        return torch.compile(fn, dynamic=False, fullgraph=True)
    except Exception:
        return fn


def _adamw_step(
    p: "torch.Tensor",
    grad: "torch.Tensor",
    exp_avg: "torch.Tensor",
    exp_avg_sq: "torch.Tensor",
    step_t: "torch.Tensor",
    lr_t: "torch.Tensor",
    beta1_t: "torch.Tensor",
    beta2_t: "torch.Tensor",
    eps_t: "torch.Tensor",
    wd_t: "torch.Tensor",
) -> None:
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / bias1).item())


def _muon_step(
    stacked_grads: "torch.Tensor",
    stacked_params: "torch.Tensor",
    momentum_buffer: "torch.Tensor",
    second_momentum_buffer: "torch.Tensor",
    momentum_t: "torch.Tensor",
    lr_t: "torch.Tensor",
    wd_t: "torch.Tensor",
    beta2_t: "torch.Tensor",
    ns_steps: "torch.Tensor",
    red_dim: int,
) -> None:
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    n_steps = int(ns_steps.item())
    if g.size(-2) > g.size(-1):
        for a, b, c in _POLAR_EXPRESS_COEFFS[:n_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in _POLAR_EXPRESS_COEFFS[:n_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X

    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# Attempt compilation for GPU performance
adamw_step_fused = _try_compile(_adamw_step)
muon_step_fused = _try_compile(_muon_step)


# ---------------------------------------------------------------------------
# MuonAdamW
# ---------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer from autoresearch-master:
      - Muon (with NorMuon variance reduction) for 2-D matrix parameters
      - AdamW for embeddings, scalars, and the lm_head

    Each param_group must include a ``kind`` key: ``"muon"`` or ``"adamw"``.

    Muon groups additionally require:
      ``momentum``, ``ns_steps``, ``beta2``, ``weight_decay``

    AdamW groups additionally require:
      ``betas``, ``eps``, ``weight_decay``
    """

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors avoid torch.compile recompilation when values change
        def _cpu(v: float = 0.0) -> "torch.Tensor":
            return torch.tensor(v, dtype=torch.float32, device="cpu")

        self._adamw_step_t = _cpu()
        self._adamw_lr_t = _cpu()
        self._adamw_beta1_t = _cpu()
        self._adamw_beta2_t = _cpu()
        self._adamw_eps_t = _cpu()
        self._adamw_wd_t = _cpu()
        self._muon_momentum_t = _cpu()
        self._muon_lr_t = _cpu()
        self._muon_wd_t = _cpu()
        self._muon_beta2_t = _cpu()

    def _step_adamw(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        self._adamw_lr_t.fill_(group["lr"])
        self._adamw_beta1_t.fill_(beta1)
        self._adamw_beta2_t.fill_(beta2)
        self._adamw_eps_t.fill_(group["eps"])
        self._adamw_wd_t.fill_(group["weight_decay"])

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            adamw_step_fused(
                p, grad,
                state["exp_avg"], state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t, self._adamw_beta1_t, self._adamw_beta2_t,
                self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict[str, Any]) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return

        self._muon_lr_t.fill_(group["lr"])
        self._muon_wd_t.fill_(group["weight_decay"])
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        ns_steps_t = torch.tensor(group["ns_steps"], dtype=torch.float32, device="cpu")

        # Stack all params of the same shape for a single fused step
        stacked_grads = torch.stack([p.grad.float() for p in params])
        stacked_params = torch.stack([p.data for p in params])

        state_key = id(group)
        state = self.state.setdefault(state_key, {})  # type: ignore[arg-type]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(stacked_grads)
            state["second_momentum_buffer"] = torch.ones_like(stacked_grads)

        red_dim = -1 if stacked_grads.size(-2) >= stacked_grads.size(-1) else -2

        muon_step_fused(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
            self._muon_beta2_t, ns_steps_t, red_dim,
        )

        for p, new_p in zip(params, stacked_params):
            p.data.copy_(new_p)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
