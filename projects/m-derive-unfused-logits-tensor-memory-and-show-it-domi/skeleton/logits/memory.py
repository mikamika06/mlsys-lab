def unfused_logits_bytes(batch_size, seq_len, vocab_size, dtype_bytes=2):
    raise NotImplementedError


def weight_memory_bytes(num_params, dtype_bytes=2):
    raise NotImplementedError


def logits_dominates_weights(batch_size, seq_len, vocab_size, num_params, dtype_bytes=2):
    raise NotImplementedError
