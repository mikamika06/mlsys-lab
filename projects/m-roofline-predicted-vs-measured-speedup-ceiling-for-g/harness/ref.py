import random

def get_test_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(5):
        flops = rng.uniform(1e8, 1e10)
        bts = rng.uniform(1e6, 1e8)
        p_flops = 1e12
        p_bw = 100e9
        base_t = rng.uniform(5.0, 20.0)
        token_f = flops
        gamma = 4

        intensity = flops / bts
        hw_intensity = p_flops / p_bw
        perf = p_flops if intensity >= hw_intensity else intensity * p_bw
        t_token = token_f / perf if perf > 0 else 0.0
        spec_total = t_token * (1.0 + gamma)
        eff_t = spec_total / (1.0 + gamma)
        ceiling = base_t / eff_t if eff_t > 0 else 1.0

        measured = ceiling * rng.uniform(0.97, 1.03)

        cases.append({
            "flops_per_token": flops,
            "bytes_per_token": bts,
            "peak_flops": p_flops,
            "peak_bandwidth": p_bw,
            "baseline_time": base_t,
            "token_flops": token_f,
            "gamma": gamma,
            "expected_intensity": intensity,
            "expected_ceiling": ceiling,
            "measured_speedup": measured
        })
    return cases
