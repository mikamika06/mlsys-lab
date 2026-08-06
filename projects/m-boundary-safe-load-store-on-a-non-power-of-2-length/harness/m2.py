import numpy as np
import ref


def check(workdir):
    from triton_bounds.analysis import compute_wasted_lane_fraction
    from triton_bounds.ops import catch_unmasked_store

    out = {"exception_caught": 0.0, "wasted_fraction_matched": 0.0}

    x = np.ones(100, dtype=np.float32)
    caught, _ = catch_unmasked_store(x, 100, block_size=64)
    if caught:
        out["exception_caught"] = 1.0
    else:
        out["_note"] = "Failed to catch unmasked store exception"

    test_pairs = [(100, 64), (128, 64), (13, 32), (300, 128), (1, 128)]
    match_count = 0
    for n, block_size in test_pairs:
        got = compute_wasted_lane_fraction(n, block_size)
        want = ref.ref_wasted_lane_fraction(n, block_size)
        if abs(got - want) < 1e-6:
            match_count += 1
        else:
            out["_note"] = f"Wasted fraction mismatch for N={n}, B={block_size}: got {got}, want {want}"
            break

    if match_count == len(test_pairs):
        out["wasted_fraction_matched"] = 1.0

    return out
