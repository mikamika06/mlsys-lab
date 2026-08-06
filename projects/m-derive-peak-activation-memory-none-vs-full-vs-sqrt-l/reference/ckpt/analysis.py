import math


def peak_activation_memory(layers, base_mem, strategy, segment_size=None):
    if strategy == "none":
        return base_mem * layers
    elif strategy == "full":
        return base_mem
    elif strategy == "sqrt":
        s = segment_size if segment_size is not None else int(math.ceil(math.sqrt(layers)))
        s = max(1, min(s, layers))
        num_segments = math.ceil(layers / s)
        return base_mem * (s + num_segments)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def optimal_segment_size(layers, base_mem):
    s = int(round(math.sqrt(layers)))
    return max(1, min(s, layers))


def recompute_flops_overhead(layers, segment_size):
    if segment_size <= 0:
        raise ValueError("Segment size must be positive")
    num_segments = math.ceil(layers / segment_size)
    recompute_ops = (num_segments - 1) * layers
    return float(recompute_ops) / float(layers)
