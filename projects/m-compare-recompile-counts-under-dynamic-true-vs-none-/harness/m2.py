import ref

def check(workdir):
    from recompiles.simulate import simulate_lru_cache
    cases = ref.get_test_cases()
    capacities = [2, 4, 8]
    ok = 0
    total = 0
    for shapes in cases:
        for cap in capacities:
            total += 1
            want = _ref_sim(shapes, cap)
            got = simulate_lru_cache(shapes, cap)
            if got == want:
                ok += 1
    return {"simulation_match": 1.0 if ok == total else 0.0}

def _ref_sim(shapes, capacity):
    cache = []
    recompiles = 0
    for shape in shapes:
        if shape in cache:
            cache.remove(shape)
            cache.append(shape)
        else:
            recompiles += 1
            if len(cache) >= capacity:
                cache.pop(0)
            cache.append(shape)
    return recompiles
