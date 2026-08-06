from hybridkv.memory import effective_bytes_per_token


def predict_startup_kv_size(config, max_seq_len, dtype_size=2):
    return effective_bytes_per_token(config, max_seq_len, dtype_size)
