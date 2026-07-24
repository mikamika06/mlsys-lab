def _oracle(L, chunk_sizes):
    out = []
    for c in chunk_sizes:
        steps = (L + c - 1) // c
        loads = []
        remaining = L
        while remaining > 0:
            take = c if remaining >= c else remaining
            loads.append(take)
            remaining -= take
        out.append({
            "chunk_size": c,
            "prefill_steps": steps,
            "step_token_loads": loads,
        })
    return out


def grade(sol, fx) -> dict:
    cases = [
        (10, [4, 6, 10]),
        (17, [1, 5, 8, 20]),
        (64, [16, 32, 7, 64]),
        (3, [1, 2, 5]),
    ]
    ok = 1.0
    for L, chunks in cases:
        try:
            got = sol.compare_chunk_knob(L, list(chunks))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(L, chunks):
            ok = 0.0
            break
    return {"exact_match": ok}
