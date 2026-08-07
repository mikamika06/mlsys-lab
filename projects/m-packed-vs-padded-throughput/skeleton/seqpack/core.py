def build_cu_seqlens(seqlens):
    raise NotImplementedError


def packed_cost(seqlens, block_size):
    raise NotImplementedError


def padded_cost(batch_size, seq_length, block_size):
    raise NotImplementedError
