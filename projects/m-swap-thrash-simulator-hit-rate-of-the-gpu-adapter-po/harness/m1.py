import ref


def check(workdir):
    from multilora.simulator import simulate_hit_rate
    reqs = [1, 2, 1, 3, 1, 2, 4, 1]
    pool = 2
    got = simulate_hit_rate(reqs, pool)
    want = ref.simulate_hit_rate(reqs, pool)
    match = 1.0 if abs(got - want) < 1e-6 else 0.0
    return {"hit_rate_match": match}
