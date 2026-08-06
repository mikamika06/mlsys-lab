import ref

def check(workdir):
    from fsdpmeasure.memory import simulate_peak_memory
    model_size = 400 * 1024 * 1024
    world = 4
    want = ref.simulate_peak_memory(model_size, world, "FULL_SHARD")
    got = simulate_peak_memory(model_size, world, "FULL_SHARD")
    match = 1.0 if len(got) == world and all(abs(a - b) < 1e-5 for a, b in zip(got, want)) else 0.0
    return {"peak_memory_match": match}
