import numpy as np
from typing import Dict, Tuple

def _oracle(lengths, bs):
    lengths_np = np.asarray(lengths, dtype=np.int64)
    max_len = int(np.max(lengths_np)) if lengths_np.size else 0
    paged_blocks = int(np.sum((lengths_np + bs - 1) // bs))
    paged_wasted = paged_blocks * bs - int(np.sum(lengths_np))

    contig_block_per_seq = (max_len + bs - 1) // bs
    contig_blocks = contig_block_per_seq * len(lengths)
    contig_wasted = contig_blocks * bs - int(np.sum(lengths_np))
    return {
        "paged": (paged_blocks, paged_wasted),
        "contiguous": (contig_blocks, contig_wasted)
    }

def grade(sol, fx) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(5):
        n_seq = rng.integers(1, 20)
        lengths = rng.integers(1, 100, size=n_seq).tolist()
        bs = int(rng.integers(4, 32))
        try:
            got = sol.wasted_slots(lengths, bs)
        except Exception:
            return {"exact_match": 0.0}
        ref = _oracle(lengths, bs)
        if not isinstance(got, dict):
            return {"exact_match": 0.0}
        for key in ("paged", "contiguous"):
            if key not in got or key not in ref:
                return {"exact_match": 0.0}
            if tuple(got[key]) != tuple(ref[key]):
                return {"exact_match": 0.0}
    return {"exact_match": ok}
