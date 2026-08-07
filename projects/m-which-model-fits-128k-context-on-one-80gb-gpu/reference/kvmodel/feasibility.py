from kvmodel.sizing import compute_kv_bytes


def fits_on_gpu(config, seq_len, batch_size, gpu_limit_bytes, dtype_bytes):
    needed = compute_kv_bytes(config, seq_len, batch_size, dtype_bytes)
    model_weights = config.get("weight_bytes", 0)
    return (needed + model_weights) <= gpu_limit_bytes
