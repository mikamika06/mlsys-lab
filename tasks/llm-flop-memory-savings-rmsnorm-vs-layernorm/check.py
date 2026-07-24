import math

def _ref(shape):
    n,d = shape
    flops_layernorm = 6*n*d + n
    mem_reads_layernorm = 6*n*d
    flops_rmsnorm = 4*n*d + n
    mem_reads_rmsnorm = 3*n*d
    return {
        "flops_layernorm": float(flops_layernorm),
        "mem_reads_layernorm": float(mem_reads_layernorm),
        "flops_rmsnorm": float(flops_rmsnorm),
        "mem_reads_rmsnorm": float(mem_reads_rmsnorm)
    }

def grade(sol, fx) -> dict:
    shapes = [(10, 20), (5, 100), (3, 4)]
    ok = 1.0
    for shape in shapes:
        try:
            got = sol.norm_flop_mem(shape)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref(shape)
        for k in ref:
            if not math.isclose(got.get(k, None), ref[k], rel_tol=1e-9):
                ok = 0.0
                break
        if ok == 0.0:
            break
    return {"exact_match": ok}
