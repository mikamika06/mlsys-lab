def get_layer_dtypes(config, keep_sliding_fp16=True):
    raise NotImplementedError


def compute_kv_bytes(config, dtypes, seq_len):
    raise NotImplementedError
