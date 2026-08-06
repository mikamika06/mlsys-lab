# Fixed latencies for L1, L2, L3 and DRAM (cycles)
LATENCIES = [1.0, 4.0, 12.0, 100.0]

def compute_amat(hit_rates: list[float]) -> float:
    """Compute AMAT from per‑level hit rates."""
    h_l1, h_l2, h_l3 = (hit_rates[0], hit_rates[1], hit_rates[2])
    m1 = 1.0 - h_l1
    m2 = 1.0 - h_l2
    m3 = 1.0 - h_l3
    return (
        LATENCIES[0]
        + m1 * (LATENCIES[1] + m2 * (LATENCIES[2] + m3 * LATENCIES[3]))
    )
