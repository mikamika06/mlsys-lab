"""Deterministic, hardware-independent scorers.

Every scorer takes plain arrays and returns a float. No timing here — speed
metrics live elsewhere and are always same-machine ratios.
"""
from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def mean_kl(ref_logits: np.ndarray, cand_logits: np.ndarray, eps: float = 1e-12) -> float:
    """Mean KL(softmax(ref) || softmax(cand)) over rows, in nats.

    The canonical quality gate for quantization: how far the candidate's output
    distribution drifted from the reference. Computed in float64 for determinism.
    """
    p = _softmax(ref_logits)
    q = _softmax(cand_logits)
    kl = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
    return float(np.mean(kl))


def argmax_agreement(ref_logits: np.ndarray, cand_logits: np.ndarray) -> float:
    """Fraction of rows whose top-1 argmax matches the reference."""
    a = np.argmax(np.asarray(ref_logits), axis=-1)
    b = np.argmax(np.asarray(cand_logits), axis=-1)
    return float(np.mean(a == b))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def size_ratio(original: np.ndarray, *quantized_parts: np.ndarray) -> float:
    """original bytes / sum of the quantized parts' ACTUAL bytes.

    Uses each part's real dtype itemsize, so returning int8 codes + fp16/fp32
    scales is rewarded and returning int64 codes is penalised automatically.
    """
    orig_bytes = np.asarray(original).nbytes
    q_bytes = sum(int(np.asarray(p).nbytes) for p in quantized_parts)
    if q_bytes == 0:
        return float("inf")
    return float(orig_bytes / q_bytes)


def rel_err(original: np.ndarray, approx: np.ndarray) -> float:
    """Global relative L2 error ``||approx - original|| / ||original||``."""
    a = np.asarray(original, dtype=np.float64).ravel()
    b = np.asarray(approx, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


def channel_rel_err(W: np.ndarray, W_hat: np.ndarray, axis: int = 1) -> float:
    """Mean per-channel relative reconstruction error, EACH CHANNEL WEIGHTED EQUALLY.

    ``mean_i  ||W_hat[i] - W[i]|| / ||W[i]||``  along ``axis``. Unlike a global MSE
    (which is dominated by high-magnitude channels), this counts a crushed quiet
    channel just as much as a loud one — so it exposes the per-tensor-vs-per-channel
    gap that magnitude-weighted metrics hide.
    """
    W = np.asarray(W, dtype=np.float64)
    W_hat = np.asarray(W_hat, dtype=np.float64)
    num = np.linalg.norm(W_hat - W, axis=axis)
    den = np.linalg.norm(W, axis=axis) + 1e-12
    return float(np.mean(num / den))


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def byte_exact_fraction(a: bytes | np.ndarray, b: bytes | np.ndarray) -> float:
    """Fraction of identical bytes between two buffers (1.0 == byte-exact)."""
    ba = a.tobytes() if isinstance(a, np.ndarray) else bytes(a)
    bb = b.tobytes() if isinstance(b, np.ndarray) else bytes(b)
    if len(ba) != len(bb) or len(ba) == 0:
        return 0.0
    same = sum(1 for x, y in zip(ba, bb) if x == y)
    return same / len(ba)
