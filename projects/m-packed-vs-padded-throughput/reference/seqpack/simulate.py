from seqpack.core import packed_cost, padded_cost

def throughput_ratio(seqlens, block_size):
    if not seqlens:
        return 1.0
    actual_max = max(seqlens)
    pck = packed_cost(seqlens, block_size)
    if pck == 0:
        return 1.0
    pad = padded_cost(len(seqlens), actual_max, block_size)
    return float(pad) / float(pck)


def misspecification_effects(seqlens, block_size, provided_max_seqlen):
    if not seqlens:
        return {"wasted_flops": 0, "relative_degradation": 1.0}
    actual_max = max(seqlens)
    if provided_max_seqlen < actual_max:
        raise ValueError("provided_max_seqlen cannot be less than actual maximum length")
    optimal_pad = padded_cost(len(seqlens), actual_max, block_size)
    actual_pad = padded_cost(len(seqlens), provided_max_seqlen, block_size)
    return {
        "wasted_flops": actual_pad - optimal_pad,
        "relative_degradation": float(actual_pad) / float(optimal_pad) if optimal_pad else 1.0
    }
