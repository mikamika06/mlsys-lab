def qkt_access_order(S: int, d: int, B: int, elem_bytes: int) -> list:
    """
    Return byte addresses in tile-blocked QK^T access order.
    Q is at base 0, K is at base S*d*elem_bytes.
    """
    addrs = []
    K_base = S * d * elem_bytes
    for ii in range(0, S, B):
        for jj in range(0, S, B):
            for i in range(ii, min(ii + B, S)):
                for j in range(jj, min(jj + B, S)):
                    for k in range(d):
                        q_addr = (i * d + k) * elem_bytes
                        k_addr = K_base + (j * d + k) * elem_bytes
                        addrs.append(q_addr)
                        addrs.append(k_addr)
    return addrs
