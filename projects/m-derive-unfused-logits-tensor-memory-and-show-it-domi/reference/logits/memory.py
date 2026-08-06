def unfused_logits_bytes(batch_size, seq_len, vocab_size, dtype_bytes=2):
    return batch_size * seq_len * vocab_size * dtype_bytes


def weight_memory_bytes(num_params, dtype_bytes=2):
    return num_params * dtype_bytes


def logits_dominates_weights(batch_size, seq_len, vocab_size, num_params, dtype_bytes=2):
    return unfused_logits_bytes(batch_size, seq_len, vocab_size, dtype_bytes) > weight_memory_bytes(num_params, dtype_bytes)
