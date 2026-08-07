def build_schedule(num_layers: int, prefetch: int) -> list[tuple[str, int]]:
    sched = []

    # Forward pass
    for i in range(min(prefetch, num_layers)):
        sched.append(("all_gather_fw", i))
    for i in range(num_layers):
        if i + prefetch < num_layers:
            sched.append(("all_gather_fw", i + prefetch))
        sched.append(("compute_fw", i))
        sched.append(("free_fw", i))

    # Backward pass
    for i in range(num_layers - 1, max(-1, num_layers - 1 - prefetch), -1):
        sched.append(("all_gather_bw", i))
    for i in range(num_layers - 1, -1, -1):
        if i - prefetch >= 0:
            sched.append(("all_gather_bw", i - prefetch))
        sched.append(("compute_bw", i))
        sched.append(("reduce_scatter", i))
        sched.append(("free_bw", i))

    return sched

def simulate_peak_memory(layers: list[int], schedule: list[tuple[str, int]]) -> int:
    active = set()
    current = 0
    peak = 0
    for op, i in schedule:
        if op in ("all_gather_fw", "all_gather_bw"):
            if i not in active:
                active.add(i)
                current += layers[i] * 2
                if current > peak:
                    peak = current
        elif op in ("free_fw", "free_bw"):
            if i in active:
                active.remove(i)
                current -= layers[i] * 2
    return peak
