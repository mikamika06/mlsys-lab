def grade(sol, fx) -> dict:
    import numpy as np
    try:
        got = sol.machine_epsilon_fp32()
    except Exception:
        return {"exact_match": 0.0}
    ref = int(np.finfo(np.float32).eps.view(np.uint32))
    ok = 1.0 if isinstance(got, (int, np.integer)) and int(got) == ref else 0.0
    return {"exact_match": ok}
