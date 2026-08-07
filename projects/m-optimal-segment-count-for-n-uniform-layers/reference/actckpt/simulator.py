from typing import Dict, Any


def simulate_checkpoint_memory(n_layers: int, num_segments: int, bytes_per_layer: int) -> Dict[str, Any]:
    """Simulates activation memory state at each step during forward and backward passes."""
    if n_layers <= 0 or num_segments <= 0:
        return {"trace": [0], "peak_bytes": 0}

    k = min(n_layers, num_segments)
    base_seg = n_layers // k
    rem = n_layers % k
    seg_sizes = [base_seg + (1 if i < rem else 0) for i in range(k)]

    trace = [0]
    current_mem = 0

    for size in seg_sizes:
        current_mem += bytes_per_layer
        trace.append(current_mem)

    for seg_idx in reversed(range(k)):
        size = seg_sizes[seg_idx]
        for step in range(1, size + 1):
            trace.append(current_mem + step * bytes_per_layer)
        for step in reversed(range(size)):
            trace.append(current_mem + step * bytes_per_layer)
        current_mem -= bytes_per_layer
        trace.append(current_mem)

    peak_bytes = max(trace)
    return {"trace": trace, "peak_bytes": peak_bytes}
