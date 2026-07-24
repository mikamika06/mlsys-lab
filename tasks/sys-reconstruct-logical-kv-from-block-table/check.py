import numpy as np
from mlsys.scorers import byte_exact_fraction

def _ref_answer(physical_store, block_table, block_size, num_valid_tokens):
    """Oracle: reconstruct logical KV via NumPy fancy indexing."""
    bt = np.asarray(block_table, dtype=np.intp)
    positions = np.arange(num_valid_tokens, dtype=np.intp)
    logical_blocks = positions // block_size
    slots = positions % block_size
    phys = bt[logical_blocks]
    return physical_store[phys, slots]

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)

    cases = [
        # (num_phys, block_size, head_dim, num_valid_tokens, block_table)
        (4, 3, 8, 7, [2, 0, 3, 1]),           # scattered, last block partial
        (3, 4, 16, 12, [1, 2, 0]),             # every slot filled
        (2, 4, 8, 1, [1, 0]),                  # single token
        (10, 2, 32, 5, [9, 3, 7, 5, 1]),       # large store, 5 logical blocks
        (1, 8, 4, 3, [0]),                      # single physical block
        (5, 3, 8, 9, [4, 2, 0]),               # 3 logical blocks, last partial
    ]

    scores = []
    for num_phys, bs, hd, num_valid, bt in cases:
        phys = rng.randn(num_phys, bs, hd).astype(np.float32)
        ref = _ref_answer(phys, bt, bs, num_valid)
        try:
            got = np.asarray(
                sol.reconstruct_logical_kv(phys, bt, bs, num_valid),
                dtype=np.float32,
            )
        except Exception:
            scores.append(0.0)
            continue
        scores.append(byte_exact_fraction(ref, got))

    return {"byte_exact_fraction": min(scores) if scores else 0.0}
