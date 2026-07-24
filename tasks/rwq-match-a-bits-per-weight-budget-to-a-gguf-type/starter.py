import numpy as np

# GGUF type definitions with a deliberate mistake in Q8_0 block size
GGUF_TYPES = [
    ("Q2_K",   16, 64),
    ("Q4_K_M", 32, 64),
    ("Q6_K",   48, 64),
    ("Q8_0",   48, 64),   # WRONG: should be 64 bytes
]

def _compute_bpw():
    block_bytes = np.array([t[1] for t in GGUF_TYPES], dtype=np.float64)
    weights     = np.array([t[2] for t in GGUF_TYPES], dtype=np.float64)
    return 8.0 * block_bytes / weights

def match_bpw(target_bpw: float) -> int:
    """Broken implementation that mis‑computes bpw for Q8_0."""
    bpw = _compute_bpw()
    diff = np.abs(bpw - float(target_bpw))
    return int(np.argmin(diff))
