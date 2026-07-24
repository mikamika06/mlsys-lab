import math, random

def _ref(peak_compute, peak_mem, compute_per_token, mem_per_token):
    theta = peak_compute / peak_mem
    val = (theta * mem_per_token) / compute_per_token
    return math.ceil(val**2)

def grade(sol, fx) -> dict:
    max_rel_err = 0.0
    for _ in range(10):
        peak_compute = random.uniform(1e9, 1e12)
        peak_mem = random.uniform(1e8, 5e10)
        compute_per_token = random.uniform(0.1, 10.0)
        mem_per_token = random.uniform(0.1, 20.0)
        ref = _ref(peak_compute, peak_mem, compute_per_token, mem_per_token)
        try:
            got = sol.crossover_batch_size(peak_compute, peak_mem,
                                            compute_per_token, mem_per_token)
        except Exception:
            return {"rel_err": 1.0}
        if not isinstance(got, int):
            return {"rel_err": 1.0}
        rel_err = abs(got - ref) / max(1, ref)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": max_rel_err}
