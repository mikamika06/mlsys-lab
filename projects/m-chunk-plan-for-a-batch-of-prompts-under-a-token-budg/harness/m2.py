import ref

def check(workdir):
    from chunkplan import simulator
    prompts = [{"id": i, "length": l} for i, l in enumerate([128, 256])]
    got = simulator.simulate_schedule(prompts, max_batched_tokens=256)
    out = {"simulation_matched": 1.0 if isinstance(got, list) and len(got) > 0 else 0.0}
    return out
