import numpy as np

def _ref(d_model, d_ff):
    # Reference calculation using NumPy for determinism
    return int(np.ceil(d_ff / 2))

def grade(sol, fx) -> dict:
    cases = [
        (768, 3072),
        (512, 2048),
        (1024, 4096),
        (256, 123),   # odd hidden size
        (128, 0),     # zero hidden size
    ]
    ok_exact = 1.0
    ok_int = 1.0
    for d_model, d_ff in cases:
        try:
            got = sol.crossover_seq_len(d_model, d_ff)
        except Exception:
            return {"exact_match": 0.0, "is_int": 0.0}
        ref = _ref(d_model, d_ff)
        if got != ref:
            ok_exact = 0.0
        if not isinstance(got, int):
            ok_int = 0.0
    return {"exact_match": ok_exact, "is_int": ok_int}
