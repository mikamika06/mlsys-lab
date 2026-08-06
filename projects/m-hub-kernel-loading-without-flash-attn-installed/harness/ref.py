class AttentionInterface:
    def __init__(self, name, available=True):
        self.name = name
        self.available = available

    def forward(self, q, k, v):
        if not self.available:
            raise RuntimeError(f"Backend {self.name} is not available.")
        return f"executed_{self.name}"


CONFIGS = [
    {
        "config": {"attn_implementation": "flash", "fallback_priority": ["flash", "sdpa", "math"]},
        "backends": [AttentionInterface("flash", False), AttentionInterface("sdpa", True), AttentionInterface("math", True)]
    },
    {
        "config": {"custom_kernel_backend": "hub_custom_flash", "fallback_priority": ["hub_custom_flash", "math"]},
        "backends": [AttentionInterface("hub_custom_flash", False), AttentionInterface("math", True)]
    },
    {
        "config": {"attn_implementation": "flash"},
        "backends": [AttentionInterface("flash", True), AttentionInterface("sdpa", True)]
    },
    {
        "config": {"attn_implementation": "non_existent", "fallback_priority": ["sdpa", "math"]},
        "backends": [AttentionInterface("sdpa", True), AttentionInterface("math", True)]
    }
]


COST_CASES = [
    {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "backends": ["flash", "sdpa", "math"]},
    {"batch_size": 4, "seq_len": 2048, "num_heads": 32, "head_dim": 128, "backends": ["flash", "sdpa"]}
]


def resolve_backend(config, available_backends):
    requested = config.get("attn_implementation") or config.get("custom_kernel_backend") or "flash"
    fallback_chain = config.get("fallback_priority", ["flash", "sdpa", "math"])

    avail_map = {b if isinstance(b, str) else b.name: (b if not isinstance(b, str) else True) for b in available_backends}

    if requested in avail_map:
        item = avail_map[requested]
        is_avail = item.available if hasattr(item, "available") else bool(item)
        if is_avail:
            return requested

    for candidate in fallback_chain:
        if candidate in avail_map:
            item = avail_map[candidate]
            is_avail = item.available if hasattr(item, "available") else bool(item)
            if is_avail:
                return candidate

    raise RuntimeError("No suitable attention backend available.")


def dispatch_attention(config, q, k, v, available_backends):
    backend_name = resolve_backend(config, available_backends)
    for b in available_backends:
        name = b if isinstance(b, str) else b.name
        if name == backend_name:
            if hasattr(b, "forward"):
                return b.forward(q, k, v)
            return f"executed_{name}"
    return f"executed_{backend_name}"


def estimate_train_step_cost(batch_size, seq_len, num_heads, head_dim, backend="sdpa"):
    flops_fwd = 4 * batch_size * num_heads * (seq_len ** 2) * head_dim
    flops_bwd = 2.5 * flops_fwd
    total_flops = flops_fwd + flops_bwd

    overhead_multipliers = {
        "flash": 1.0,
        "sdpa": 1.25,
        "math": 2.5,
    }
    multiplier = overhead_multipliers.get(backend, 2.0)

    memory_bytes = batch_size * num_heads * seq_len * head_dim * 2 * 3
    if backend == "math":
        memory_bytes += batch_size * num_heads * (seq_len ** 2) * 4

    estimated_cost = total_flops * multiplier + memory_bytes
    return {
        "flops": total_flops,
        "memory_bytes": memory_bytes,
        "effective_cost": estimated_cost
    }


def compare_backend_costs(batch_size, seq_len, num_heads, head_dim, available_backends):
    results = {}
    for b in available_backends:
        name = b if isinstance(b, str) else b.name
        results[name] = estimate_train_step_cost(batch_size, seq_len, num_heads, head_dim, backend=name)
    return results
