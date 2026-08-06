def chunked_crossentropy_bytes(batch_size, seq_len, vocab_size, chunk_size, dtype_bytes=2):
    num_chunks = (vocab_size + chunk_size - 1) // chunk_size
    return batch_size * seq_len * chunk_size * dtype_bytes + num_chunks * 64


def memory_savings_ratio(batch_size, seq_len, vocab_size, chunk_size, dtype_bytes=2):
    full = batch_size * seq_len * vocab_size * dtype_bytes
    chunked = chunked_crossentropy_bytes(batch_size, seq_len, vocab_size, chunk_size, dtype_bytes)
    return full / float(chunked)
