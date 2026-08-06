BUDGET_BYTES = 36 * 1024 * 1024 * 1024

CONFIGS = [
    {
        "name": "llama-70b-mha",
        "type": "mha",
        "num_layers": 80,
        "num_kv_heads": 8,
        "num_attention_heads": 64,
        "head_dim": 128,
        "bytes_per_elem": 2,
    },
    {
        "name": "llama-70b-gqa",
        "type": "gqa",
        "num_layers": 80,
        "num_kv_heads": 8,
        "num_attention_heads": 64,
        "head_dim": 128,
        "bytes_per_elem": 2,
    },
    {
        "name": "deepseek-mla",
        "type": "mla",
        "num_layers": 61,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "bytes_per_elem": 2,
    },
]


def compute_kv_bytes(config, context_length, batch_size=1):
    t = config["type"]
    L = config["num_layers"]
    b = batch_size
    s = context_length
    bytes_e = config["bytes_per_elem"]

    if t in ("mha", "gqa"):
        kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        # K and V cache: 2 * L * b * s * kv_heads * head_dim * bytes_per_elem
        return 2 * L * b * s * kv_heads * head_dim * bytes_e
    elif t == "mla":
        # MLA compresses KV into latent vector kv_lora_rank plus decoupled rope key head dim
        latent_dim = config["kv_lora_rank"]
        rope_dim = config["qk_rope_head_dim"]
        # Cache stores compressed latent cache + rope cache per layer
        return L * b * s * (latent_dim + rope_dim) * bytes_e
    else:
        raise ValueError(f"Unknown type {t}")


def max_context_length(config, budget_bytes, batch_size=1):
    low = 1
    high = 10_000_000
    best = 0
    while low <= high:
        mid = (low + high) // 2
        needed = compute_kv_bytes(config, mid, batch_size)
        if needed <= budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def back_calculate_oom(config, failed_context_length, batch_size=1):
    failed_bytes = compute_kv_bytes(config, failed_context_length, batch_size)
    if failed_bytes <= BUDGET_BYTES:
        raise ValueError("Did not actually OOM under budget")
    # Scale down proportionally to the budget
    max_len = max_context_length(config, BUDGET_BYTES, batch_size)
    return max_len
