class AttentionInterface:
    def __init__(self, name, available=True):
        self.name = name
        self.available = available

    def forward(self, q, k, v):
        if not self.available:
            raise RuntimeError(f"Backend {self.name} is not available.")
        return f"executed_{self.name}"


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
