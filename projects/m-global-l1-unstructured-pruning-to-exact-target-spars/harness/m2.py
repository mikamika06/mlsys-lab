import ref

def check(workdir):
    from prune.speedup import measure_cpu_speedup
    weights = ref.get_test_weights()
    got = measure_cpu_speedup(weights, 0.9)
    want = ref.measure_cpu_speedup(weights, 0.9)
    match = 1.0 if abs(got - want) < 0.2 else 0.0
    return {"speedup_match": match}
