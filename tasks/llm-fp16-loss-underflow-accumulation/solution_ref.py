import numpy as np


def per_token_ce(logits16: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-token cross-entropy from float16 logits, computed in float32."""
    z = np.asarray(logits16, dtype=np.float32)          # widen BEFORE any exp()
    y = np.asarray(targets, dtype=np.int64)
    n = z.shape[0]
    m = np.max(z, axis=1)                               # max shift keeps exp() in range
    lse = m + np.log(np.sum(np.exp(z - m[:, None]), axis=1, dtype=np.float32))
    return (lse.astype(np.float32) - z[np.arange(n), y]).astype(np.float32)


def mean_ce_fp32(logits16: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy with a float32 (wide) accumulator."""
    per_tok = per_token_ce(logits16, targets)
    total = np.sum(per_tok, dtype=np.float32)
    return float(total) / float(per_tok.shape[0])


def fp16_accum_stall_index(losses: np.ndarray) -> int:
    """First index whose float16 contribution is swallowed by the running sum."""
    acc = np.float16(0.0)
    zero16 = np.float16(0.0)
    for i, x in enumerate(np.asarray(losses, dtype=np.float32)):
        l16 = np.float16(x)
        new = np.float16(acc + l16)
        if l16 != zero16 and new == acc:
            return i
        acc = new
    return -1
