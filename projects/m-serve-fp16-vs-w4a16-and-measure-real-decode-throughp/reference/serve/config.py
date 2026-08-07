def make_config(model_name: str, quant_format: str, batch_size: int, seq_len: int):
    return {
        "model_name": model_name,
        "quant_format": quant_format,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }
