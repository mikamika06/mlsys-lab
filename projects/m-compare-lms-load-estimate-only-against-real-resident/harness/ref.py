import math

KV_BYTES_PER_ELEMENT = 2
PAGE_SIZE = 4096
SCRATCH_TOKENS = 256
ACTIVATION_BYTES_PER_ELEMENT = 4
RUNTIME_OVERHEAD_BYTES = 64 * 1024 * 1024


def param_count(config):
    h = config["hidden_size"]
    kv_h = config["n_kv_heads"]
    hd = config["head_dim"]
    inter = config["intermediate_size"]
    vocab = config["vocab_size"]
    n = config["n_layers"]
    embed = vocab * h
    per_layer = (h * h) + 2 * (h * kv_h * hd) + (h * h) + 3 * (h * inter) + 2 * h
    total = embed + n * per_layer + h
    if not config.get("tie_embeddings", False):
        total += vocab * h
    return total


def _round_up(n, block):
    return ((n + block - 1) // block) * block


def kv_cache_bytes(config, context_length):
    return (config["n_layers"] * 2 * config["n_kv_heads"] * config["head_dim"]
            * context_length * KV_BYTES_PER_ELEMENT)


def estimate_bytes(config, context_length, bytes_per_param):
    weight_bytes = int(math.ceil(param_count(config) * bytes_per_param))
    return weight_bytes + kv_cache_bytes(config, context_length)


def resident_bytes(config, context_length, bytes_per_param, page_size=PAGE_SIZE):
    weight_bytes = int(math.ceil(param_count(config) * bytes_per_param))
    kv_bytes = kv_cache_bytes(config, context_length)
    scratch = config["n_layers"] * SCRATCH_TOKENS * config["intermediate_size"] * ACTIVATION_BYTES_PER_ELEMENT
    return (_round_up(weight_bytes, page_size) + _round_up(kv_bytes, page_size)
            + scratch + RUNTIME_OVERHEAD_BYTES)


def relative_error(estimate, resident):
    if resident == 0:
        return 0.0
    return abs(resident - estimate) / resident


def offload_split(total_bytes, gpu_ratio, page_size=PAGE_SIZE):
    raw_gpu = int(total_bytes * gpu_ratio)
    gpu_bytes = min((raw_gpu // page_size) * page_size, total_bytes)
    return {"gpu_bytes": gpu_bytes, "cpu_bytes": total_bytes - gpu_bytes}


CONFIGS = [
    {"n_layers": 6, "hidden_size": 512, "n_heads": 8, "n_kv_heads": 8, "head_dim": 64,
     "intermediate_size": 1376, "vocab_size": 32000, "tie_embeddings": True},
    {"n_layers": 16, "hidden_size": 2048, "n_heads": 16, "n_kv_heads": 4, "head_dim": 128,
     "intermediate_size": 5632, "vocab_size": 32000, "tie_embeddings": False},
    {"n_layers": 28, "hidden_size": 3072, "n_heads": 24, "n_kv_heads": 8, "head_dim": 128,
     "intermediate_size": 8192, "vocab_size": 49152, "tie_embeddings": False},
]

CONTEXTS = [2048, 8192]
BYTES_PER_PARAM = [2.0, 0.5]

CASES = [(cfg, ctx, bpp) for cfg in CONFIGS for ctx in CONTEXTS for bpp in BYTES_PER_PARAM]
