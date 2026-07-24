"""Grader for `llm-fp16-loss-underflow-accumulation`.

Everything the candidate is compared against is recomputed here with NumPy:
 * the per-token cross-entropy oracle runs the same max-shifted log-sum-exp in
   float64 on the *same* float16 logits (so the only difference that can show
   up is the working precision of the candidate);
 * the stall oracle runs real IEEE binary16 arithmetic through numpy scalars.
No expected value is ever hardcoded.
"""
from __future__ import annotations

import numpy as np

from mlsys import scorers  # noqa: F401  (scorers.max_abs_err used below)


# --------------------------------------------------------------------------- data
def _logit_cases():
    """Deterministic (logits_fp16, targets) pairs.

    Case A: a confident model -> tiny per-token losses; their fp16 running sum
            stalls long before the end of the sequence.
    Case B: wide, high-magnitude logits -> exp() overflows in fp16 and even a
            naive fp32 softmax without the max shift blows up.
    """
    out = []

    rng = np.random.default_rng(0)
    N, V = 20000, 32
    z = rng.standard_normal((N, V)).astype(np.float32)
    t = rng.integers(0, V, size=N).astype(np.int64)
    z[np.arange(N), t] += np.float32(7.5)          # confident -> CE ~ 1e-2
    out.append((z.astype(np.float16), t))

    rng = np.random.default_rng(1)
    N, V = 6000, 128
    z = (rng.standard_normal((N, V)) * 18.0).astype(np.float32)
    t = rng.integers(0, V, size=N).astype(np.int64)
    out.append((z.astype(np.float16), t))

    return out


def _loss_cases():
    """Deterministic float32 loss vectors for the fp16 accumulator probe."""
    out = []

    rng = np.random.default_rng(2)
    out.append((rng.random(6000).astype(np.float32) * np.float32(0.05)
                + np.float32(0.002)).astype(np.float32))

    rng = np.random.default_rng(3)
    v = (rng.random(4000).astype(np.float32) * np.float32(0.3)).astype(np.float32)
    v[:50] = np.float32(0.0)                        # exact zeros must NOT count as a stall
    out.append(v)

    # short and loud: the fp16 accumulator never absorbs anything -> -1
    out.append(np.full(12, 1.5, dtype=np.float32))

    rng = np.random.default_rng(4)
    out.append((rng.random(15000).astype(np.float32) * np.float32(0.004)
                + np.float32(1e-4)).astype(np.float32))

    return out


# --------------------------------------------------------------------------- oracles
def _oracle_per_token(logits16: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """float64 max-shifted log-sum-exp cross-entropy on the given fp16 logits."""
    z = np.asarray(logits16, dtype=np.float64)
    n = z.shape[0]
    m = np.max(z, axis=1)
    lse = m + np.log(np.sum(np.exp(z - m[:, None]), axis=1))
    return lse - z[np.arange(n), np.asarray(targets, dtype=np.int64)]


def _oracle_stall(losses: np.ndarray) -> int:
    """Real binary16 sequential accumulation; index of the first absorbed term."""
    acc = np.float16(0.0)
    for i, x in enumerate(np.asarray(losses, dtype=np.float32)):
        l16 = np.float16(x)
        new = np.float16(acc + l16)
        if l16 != np.float16(0.0) and new == acc:
            return i
        acc = new
    return -1


# --------------------------------------------------------------------------- grade
def grade(sol, fx) -> dict:
    max_abs_err = 0.0
    mean_loss_abs_err = 0.0
    stall_exact = 1.0

    for logits16, targets in _logit_cases():
        truth = _oracle_per_token(logits16, targets)

        try:
            got = np.asarray(sol.per_token_ce(logits16.copy(), targets.copy()),
                             dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"),
                    "mean_loss_abs_err": float("inf"),
                    "stall_exact": 0.0}
        if got.shape != truth.shape or not np.all(np.isfinite(got)):
            max_abs_err = float("inf")
        else:
            max_abs_err = max(max_abs_err, scorers.max_abs_err(truth, got))

        try:
            got_mean = float(sol.mean_ce_fp32(logits16.copy(), targets.copy()))
        except Exception:
            return {"max_abs_err": float(max_abs_err),
                    "mean_loss_abs_err": float("inf"),
                    "stall_exact": 0.0}
        if not np.isfinite(got_mean):
            mean_loss_abs_err = float("inf")
        else:
            mean_loss_abs_err = max(mean_loss_abs_err,
                                    abs(got_mean - float(np.mean(truth))))

    for losses in _loss_cases():
        want = _oracle_stall(losses)
        try:
            have = int(sol.fp16_accum_stall_index(losses.copy()))
        except Exception:
            stall_exact = 0.0
            break
        if have != want:
            stall_exact = 0.0
            break

    return {
        "max_abs_err": float(max_abs_err),
        "mean_loss_abs_err": float(mean_loss_abs_err),
        "stall_exact": float(stall_exact),
    }
