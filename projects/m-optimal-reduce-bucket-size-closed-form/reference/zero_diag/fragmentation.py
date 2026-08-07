import numpy as np

def compute_fragmentation_curve(param_shapes, elem_bytes=2, alignment_bytes=512):
    """
    Computes cumulative memory allocation and fragmentation when contiguous_gradients=False.
    """
    exact_bytes = []
    padded_bytes = []

    cum_exact = 0
    cum_padded = 0

    for shape in param_shapes:
        numel = int(np.prod(shape))
        raw_b = numel * elem_bytes
        pad_b = int(np.ceil(raw_b / alignment_bytes) * alignment_bytes)

        cum_exact += raw_b
        cum_padded += pad_b

        exact_bytes.append(cum_exact)
        padded_bytes.append(cum_padded)

    frag_ratios = [
        (p - e) / float(p) if p > 0 else 0.0
        for e, p in zip(exact_bytes, padded_bytes)
    ]

    return {
        "cumulative_exact_bytes": exact_bytes,
        "cumulative_padded_bytes": padded_bytes,
        "fragmentation_ratio_curve": frag_ratios,
        "total_overhead_bytes": cum_padded - cum_exact
    }
