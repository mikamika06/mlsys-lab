def baseline(n_layers: int, layer_mem: int, fwd_time: int, bwd_time: int) -> dict:
    return {
        "peak_mem": n_layers * layer_mem,
        "step_time": n_layers * (fwd_time + bwd_time)
    }


def simulate_checkpointing(n_layers: int, segments: int, layer_mem: int, fwd_time: int, bwd_time: int) -> dict:
    base = n_layers // segments
    rem = n_layers % segments
    sizes = [base + 1] * rem + [base] * (segments - rem)

    mem = 0
    peak = 0
    time_total = 0

    for s_size in sizes:
        mem += layer_mem
        peak = max(peak, mem)
        time_total += s_size * fwd_time

    for s_size in reversed(sizes):
        mem += (s_size - 1) * layer_mem
        peak = max(peak, mem)
        time_total += s_size * fwd_time

        for _ in range(s_size):
            time_total += bwd_time
            mem -= layer_mem

    return {"peak_mem": peak, "step_time": time_total}


def optimal_segments(n_layers: int) -> int:
    best_s = 1
    min_peak = float('inf')
    for s in range(1, n_layers + 1):
        peak = simulate_checkpointing(n_layers, s, 1, 1, 1)["peak_mem"]
        if peak < min_peak:
            min_peak = peak
            best_s = s
    return best_s
