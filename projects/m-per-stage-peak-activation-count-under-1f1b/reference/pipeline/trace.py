def generate_memory_trace(num_stages, num_microbatches, stage, bytes_per_act):
    peak = min(num_microbatches, num_stages - stage + 2)
    trace = []
    for step in range(num_stages + num_microbatches * 2):
        active = max(0, min(peak, step + 1))
        trace.append(active * bytes_per_act)
    return trace
