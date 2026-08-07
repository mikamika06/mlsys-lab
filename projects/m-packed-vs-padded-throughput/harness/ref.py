import math

def build_cu_seqlens(seqlens):
    out = [0]
    for s in seqlens:
        out.append(out[-1] + s)
    return out


def packed_cost(seqlens, block_size):
    return sum((math.ceil(s / block_size) * block_size) ** 2 for s in seqlens)


def padded_cost(batch_size, seq_length, block_size):
    return batch_size * (math.ceil(seq_length / block_size) * block_size) ** 2


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


FIXTURES = [
    ([], 16, 0),
    ([10, 20, 30], 16, 32),
    ([10, 20, 30], 16, 64),
    ([5, 5, 5, 5], 32, 5),
    ([128, 256, 12, 1024, 8], 128, 2048),
    ([1000, 1000], 64, 2048)
]
