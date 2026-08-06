import numpy as np
from attnparity.padding import compute_attention


def assert_backend_parity(q, k, v, mask=None, backends=("eager", "sdpa"), atol=1e-4, is_causal=False):
    """Checks tensor output parity across attention backends."""
    if not backends:
        raise ValueError("At least one backend must be provided.")

    ref_backend = backends[0]
    ref_out = compute_attention(q, k, v, mask=mask, backend=ref_backend, is_causal=is_causal)

    max_diffs = {}
    all_passed = True

    for b in backends[1:]:
        out = compute_attention(q, k, v, mask=mask, backend=b, is_causal=is_causal)
        diff = float(np.max(np.abs(ref_out - out)))
        pair_key = f"{ref_backend}_vs_{b}"
        max_diffs[pair_key] = diff
        if diff > atol:
            all_passed = False

    return {
        "parity_passed": all_passed,
        "max_diffs": max_diffs,
    }
