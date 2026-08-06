def align_chunks(chunk_size, block_size):
    remainder = chunk_size % block_size
    if remainder == 0:
        return chunk_size
    return chunk_size + (block_size - remainder)
