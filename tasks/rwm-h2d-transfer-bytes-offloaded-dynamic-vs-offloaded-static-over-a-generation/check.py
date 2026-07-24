def _oracle(L, layer_bytes, T, max_len):
    """Step-by-step loop oracle — intentionally avoids the closed form."""
    if L + T > max_len:
        raise ValueError("L + T exceeds max_len")
    dynamic = 0
    for t in range(T):
        dynamic += (L + t) * layer_bytes
    static = L * layer_bytes          # initial prefill transfer
    for _ in range(T):
        static += layer_bytes         # one new entry per decode step
    return (dynamic, static)

def grade(sol, fx) -> dict:
    cases = [
        # (L, layer_bytes, T, max_len)
        (128, 4096, 100, 256),
        (1, 1, 1, 2),
        (1024, 8192, 512, 2048),
        (100, 256, 0, 100),       # T == 0 edge case
        (50, 1024, 50, 100),
        (7, 3, 10, 20),
        (1000, 16, 1, 1001),
    ]
    error_case = (100, 256, 50, 100)  # L + T > max_len → must raise ValueError

    ok = 1.0

    for L, lb, T, ml in cases:
        try:
            got = sol.h2d_transfer_bytes(L, lb, T, ml)
        except Exception:
            ok = 0.0
            break
        ref = _oracle(L, lb, T, ml)
        if got != ref:
            ok = 0.0
            break

    if ok == 1.0:
        try:
            sol.h2d_transfer_bytes(*error_case)
            ok = 0.0          # should have raised ValueError
        except ValueError:
            pass              # correct
        except Exception:
            ok = 0.0          # wrong exception type

    return {"exact_match": ok}
