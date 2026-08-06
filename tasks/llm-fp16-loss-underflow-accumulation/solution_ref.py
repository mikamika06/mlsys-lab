import math
import numpy as np


def per_token_ce(logits16: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-token cross-entropy from float16 logits, computed in float32."""
    n, v = logits16.shape
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        m = np.float32(logits16[i, 0])
        for j in range(1, v):
            val = np.float32(logits16[i, j])
            if val > m:
                m = val

        s = np.float32(0.0)
        for j in range(v):
            diff = np.float32(np.float32(logits16[i, j]) - m)
            s = np.float32(s + np.float32(math.exp(diff)))

        lse = np.float32(m + np.float32(math.log(s)))
        y = int(targets[i])
        out[i] = np.float32(lse - np.float32(logits16[i, y]))

    return out


def mean_ce_fp32(logits16: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy with a float32 (wide) accumulator."""
    per_tok = per_token_ce(logits16, targets)
    total = np.float32(0.0)
    for i in range(len(per_tok)):
        total = np.float32(total + per_tok[i])
    return float(total) / float(len(per_tok))


def fp16_accum_stall_index(losses: np.ndarray) -> int:
    """First index whose float16 contribution is swallowed by the running sum."""
    acc = np.float16(0.0)
    zero16 = np.float16(0.0)
    for i in range(len(losses)):
        l16 = np.float16(np.float32(losses[i]))
        new = np.float16(acc + l16)
        if l16 != zero16 and new == acc:
            return i
        acc = new
    return -1
