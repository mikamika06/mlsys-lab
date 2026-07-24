def simulate_activation_write(n: int, store_type: str) -> list[int]:
    """
    Emit deterministic byte addresses modeling activation write stream.
    """
    line_bytes = 64
    addr_per_float = 4
    addrs = []
    seen_lines = set()
    for i in range(n):
        addr = i * addr_per_float
        line_base = (addr // line_bytes) * line_bytes
        if store_type == "temporal":
            # read-for-ownership for this new line
            if line_base not in seen_lines:
                seen_lines.add(line_base)
                # add read accesses of that line
                for j in range(0, line_bytes, addr_per_float):
                    addrs.append(line_base + j)
        # always add the write itself
        addrs.append(addr)
    return addrs
