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
