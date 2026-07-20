"""
Training constants ported from autoresearch-master/prepare.py.

These are fixed parameters — do not modify them in experiments.
The evaluation harness reads these to ensure fair cross-experiment comparison.
"""
from __future__ import annotations

import os

# Fixed evaluation budget (bits-per-byte is computed over this many tokens)
MAX_SEQ_LEN: int = 2048
TIME_BUDGET: int = 300          # wall-clock training seconds (5 minutes)
EVAL_TOKENS: int = 40 * 524288  # ~20M tokens for val eval

# Dataset
VOCAB_SIZE: int = 8192
MAX_SHARD: int = 6542
VAL_SHARD: int = MAX_SHARD
VAL_FILENAME: str = f"shard_{VAL_SHARD:05d}.parquet"
BASE_URL: str = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
)
SPLIT_PATTERN: str = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?"""
    r"""[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
SPECIAL_TOKENS: list[str] = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN: str = "<|reserved_0|>"

# Local cache
CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR: str = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR: str = os.path.join(CACHE_DIR, "tokenizer")

# Training defaults (may be overridden by ExperimentLoop)
ASPECT_RATIO: int = 64
HEAD_DIM: int = 128
WINDOW_PATTERN: str = "SSSL"
TOTAL_BATCH_SIZE: int = 2 ** 19     # ~524K tokens per step
EMBEDDING_LR: float = 0.6
UNEMBEDDING_LR: float = 0.004
MATRIX_LR: float = 0.04
SCALAR_LR: float = 0.5
WEIGHT_DECAY: float = 0.2
ADAM_BETAS: tuple[float, float] = (0.8, 0.95)
WARMUP_RATIO: float = 0.0
WARMDOWN_RATIO: float = 0.5
FINAL_LR_FRAC: float = 0.0
DEFAULT_DEPTH: int = 8
DEVICE_BATCH_SIZE: int = 128
