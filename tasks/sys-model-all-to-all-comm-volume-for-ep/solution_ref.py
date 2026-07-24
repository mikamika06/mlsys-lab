def moe_ep_comm_bytes(T, k, d, N, bytes_per_elem=2):
    phase_bytes = T * k * d * bytes_per_elem
    return int(2 * phase_bytes)
