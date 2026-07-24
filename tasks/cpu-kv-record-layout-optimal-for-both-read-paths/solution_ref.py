def kv_record_layout_trace(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    elem_bytes: int,
    base_addr: int = 0,
) -> dict:
    def addr(t, h, k, d):
        index = (((t * num_heads + h) * 2 + k) * head_dim + d)
        return base_addr + index * elem_bytes

    newest = num_tokens - 1

    write_addrs = []
    for h in range(num_heads):
        for k in range(2):
            for d in range(head_dim):
                write_addrs.append(addr(newest, h, k, d))

    read_addrs = []
    for h in range(num_heads):
        for t in range(num_tokens):
            for k in range(2):
                for d in range(head_dim):
                    read_addrs.append(addr(t, h, k, d))

    return {
        "layout_id": "THKD",
        "write_addrs": write_addrs,
        "read_addrs": read_addrs,
    }
