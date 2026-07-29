from .groups import build_groups


def _blocks(tokens, block_size):
    return (tokens + block_size - 1) // block_size


def group_bytes(group, max_context, block_size, bytes_per_element):
    span = max_context if group["kind"] == "full" else min(group["window"], max_context)
    per_token = 2 * group["kv_heads"] * group["head_dim"] * bytes_per_element
    return _blocks(span, block_size) * block_size * per_token * len(group["layers"])


def plan_bytes(config, max_context, block_size, bytes_per_element):
    return sum(group_bytes(g, max_context, block_size, bytes_per_element)
               for g in build_groups(config))


def uniform_bytes(config, max_context, block_size, bytes_per_element):
    total = 0
    for layer in config["layers"]:
        per_token = 2 * layer["kv_heads"] * layer["head_dim"] * bytes_per_element
        total += _blocks(max_context, block_size) * block_size * per_token
    return total
