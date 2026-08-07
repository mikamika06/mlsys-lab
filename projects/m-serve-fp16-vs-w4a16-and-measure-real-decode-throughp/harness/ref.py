CONFIGS = [
    {"model_name": "llama-7b", "quant_format": "fp16", "batch_size": 4, "seq_len": 512},
    {"model_name": "llama-7b", "quant_format": "w4a16", "batch_size": 4, "seq_len": 512},
    {"model_name": "llama-70b", "quant_format": "w4a16", "batch_size": 16, "seq_len": 1024},
]


def make_config(model_name: str, quant_format: str, batch_size: int, seq_len: int):
    return {
        "model_name": model_name,
        "quant_format": quant_format,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def compute_throughput_ratio(fp16_tokens: float, w4a16_tokens: float) -> float:
    if fp16_tokens <= 0:
        return 0.0
    return float(w4a16_tokens / fp16_tokens)


def compute_memory_delta(fp16_bytes: int, w4a16_bytes: int) -> float:
    return float(fp16_bytes - w4a16_bytes)


def simulate_decode_step(config: dict) -> dict:
    bs = config["batch_size"]
    q = config["quant_format"]
    base_tok = float(bs * 45.0)
    base_mem = int(bs * 1024 * 1024 * 512)
    if q == "w4a16":
        tok = base_tok * 1.35
        mem = int(base_mem * 0.55)
    else:
        tok = base_tok
        mem = base_mem
    return {"tokens_per_sec": tok, "memory_bytes": mem}
