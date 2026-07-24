def compare_chunk_knob(L, chunk_sizes):
    result = []
    for c in chunk_sizes:
        loads = []
        remaining = L
        while remaining > 0:
            load = min(c, remaining)
            loads.append(load)
            remaining -= load
        result.append({
            "chunk_size": c,
            "prefill_steps": len(loads),
            "step_token_loads": loads,
        })
    return result
