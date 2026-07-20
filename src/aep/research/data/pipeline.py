"""
Data pipeline ported from autoresearch-master/prepare.py.

Provides:
  - download_data()     — fetch parquet shards from HuggingFace
  - train_tokenizer()   — train a BPE tokenizer with rustbpe
  - Tokenizer           — lightweight wrapper around the trained tokenizer
  - make_dataloader()   — BOS-aligned best-fit packed dataloader
  - evaluate_bpb()      — fixed evaluation metric (bits per byte, lower=better)

Heavy dependencies (torch, rustbpe, tiktoken, pyarrow, requests) are imported
lazily so that the rest of AEP can import this module without them installed.
The ``PREPARE_DEPS_AVAILABLE`` flag indicates whether training is possible.
"""
from __future__ import annotations

import math
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Generator, Iterator

from aep.research.data.constants import (
    BASE_URL,
    BOS_TOKEN,
    CACHE_DIR,
    DATA_DIR,
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    MAX_SHARD,
    SPECIAL_TOKENS,
    SPLIT_PATTERN,
    TOKENIZER_DIR,
    VAL_FILENAME,
    VAL_SHARD,
    VOCAB_SIZE,
)

# ---------------------------------------------------------------------------
# Optional heavy deps
# ---------------------------------------------------------------------------

PREPARE_DEPS_AVAILABLE = False
try:
    import requests as _requests
    import pyarrow.parquet as _pq
    import rustbpe as _rustbpe  # type: ignore[import-untyped]
    import tiktoken as _tiktoken  # type: ignore[import-untyped]
    import torch as _torch
    PREPARE_DEPS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_single_shard(index: int) -> bool:
    """Download one parquet shard with retries. Returns True on success."""
    import requests  # noqa: PLC0415

    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True
    url = f"{BASE_URL}/{filename}"
    for attempt in range(1, 6):
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            tmp = filepath + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(tmp, filepath)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  Attempt {attempt}/5 failed for {filename}: {exc}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < 5:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards: int = 100, download_workers: int = 8) -> None:
    """Download training shards and the pinned validation shard."""
    if not PREPARE_DEPS_AVAILABLE:
        raise RuntimeError("Install requests, pyarrow to use download_data()")
    os.makedirs(DATA_DIR, exist_ok=True)
    ids = list(range(min(num_shards, MAX_SHARD)))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards present at {DATA_DIR}")
        return
    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already present)…")
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(_download_single_shard, ids)
    print(f"Data: {sum(results)}/{len(ids)} shards ready at {DATA_DIR}")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _list_parquet_files() -> list[str]:
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and ".tmp" not in f)
    return [os.path.join(DATA_DIR, f) for f in files]


def _text_iterator(max_chars: int = 1_000_000_000, doc_cap: int = 10_000) -> Generator[str, None, None]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    parquet_paths = [p for p in _list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap]
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer() -> None:
    """Train a BPE tokenizer using rustbpe and save as a tiktoken-compatible pickle."""
    if not PREPARE_DEPS_AVAILABLE:
        raise RuntimeError("Install rustbpe, tiktoken to use train_tokenizer()")
    import rustbpe  # noqa: PLC0415
    import tiktoken  # noqa: PLC0415
    import pickle  # noqa: PLC0415

    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    print("Tokenizer: training…")
    bpe = rustbpe.BPE()
    bpe.train(
        texts=list(_text_iterator(max_chars=100_000_000)),
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        split_pattern=SPLIT_PATTERN,
    )
    enc = tiktoken.Encoding(
        name="autoresearch",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=bpe.get_merges(),
        special_tokens={t: i for i, t in enumerate(SPECIAL_TOKENS)},
    )
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    import torch  # noqa: PLC0415
    token_bytes_list = [enc.decode_single_token_bytes(i) for i in range(enc.n_vocab)]
    torch.save(token_bytes_list, token_bytes_path)
    print(f"Tokenizer: saved (vocab_size={enc.n_vocab})")


class Tokenizer:
    """Minimal tokenizer wrapper around a trained tiktoken encoding."""

    def __init__(self, enc: Any) -> None:
        self.enc = enc
        self.bos_token_id: int = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir: str = TOKENIZER_DIR) -> "Tokenizer":
        if not PREPARE_DEPS_AVAILABLE:
            raise RuntimeError("Install tiktoken to load the tokenizer")
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)  # noqa: S301
        return cls(enc)

    def get_vocab_size(self) -> int:
        return int(self.enc.n_vocab)

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text: str | list[str], prepend: int | str | None = None) -> Any:
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)  # type: ignore[possibly-undefined]
        else:
            ids = self.enc.encode_ordinary_batch(text, num_threads=8)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)  # type: ignore[possibly-undefined]
        return ids

    def decode(self, ids: list[int]) -> str:
        return str(self.enc.decode(ids))


def get_token_bytes(device: str = "cpu") -> Any:
    import torch  # noqa: PLC0415
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def _document_batches(
    split: str, tokenizer_batch_size: int = 128
) -> Generator[tuple[list[str], int], None, None]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    parquet_paths = _list_parquet_files()
    assert parquet_paths, "No parquet files found — run download_data() first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert parquet_paths, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    buffer_size: int = 1000,
) -> Iterator[tuple[Any, Any, int]]:
    """
    BOS-aligned dataloader with best-fit packing (100% utilisation).
    Yields (inputs, targets, epoch) tensors on CUDA.
    """
    import torch  # noqa: PLC0415

    assert split in ("train", "val")
    row_capacity = seq_len + 1
    batches = _document_batches(split)
    bos = tokenizer.get_bos_token_id()
    doc_buffer: list[list[int]] = []
    epoch = 1

    def refill() -> None:
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[: batch_size * seq_len].view(batch_size, seq_len)
    cpu_targets = cpu_buffer[batch_size * seq_len :].view(batch_size, seq_len)
    inputs = gpu_buffer[: batch_size * seq_len].view(batch_size, seq_len)
    targets = gpu_buffer[batch_size * seq_len :].view(batch_size, seq_len)

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    dl = len(doc)
                    if dl <= remaining and dl > best_len:
                        best_idx, best_len = i, dl
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (fixed metric — do not change)
# ---------------------------------------------------------------------------

@staticmethod  # type: ignore[misc]
def evaluate_bpb(model: Any, tokenizer: Tokenizer, batch_size: int) -> float:
    """
    Bits per byte (BPB) — vocab-size-independent evaluation metric.

    Lower is better. Computed over EVAL_TOKENS tokens from the pinned
    validation shard. This function is the ground-truth metric — do not
    modify it in experiments.
    """
    import torch  # noqa: PLC0415

    model.eval()
    token_bytes = get_token_bytes(device=next(model.parameters()).device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    total_nats = 0.0
    total_bytes = 0
    tokens_seen = 0

    with torch.no_grad():
        while tokens_seen < EVAL_TOKENS:
            x, y, _ = next(val_loader)
            loss_per_token = model(x, y, reduction="none").view(-1)
            y_flat = y.view(-1)
            for token_id, loss_val in zip(y_flat.tolist(), loss_per_token.tolist()):
                nbytes = len(token_bytes[token_id])
                if nbytes == 0:
                    continue  # skip special tokens
                total_nats += loss_val
                total_bytes += nbytes
            tokens_seen += x.numel()

    model.train()
    bits_per_byte = (total_nats / total_bytes) / math.log(2)
    return bits_per_byte
