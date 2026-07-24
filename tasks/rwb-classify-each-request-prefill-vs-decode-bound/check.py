import numpy as np

def _oracle(requests):
    """Compute reference labels using the official rule."""
    p = np.asarray([r["prompt_len"] for r in requests], dtype=np.int64)
    g = np.asarray([r["gen_len"]   for r in requests], dtype=np.int64)
    prefill_cost = p ** 2
    decode_cost  = g * (p + g)
    return ["prefill" if pc >= dc else "decode"
            for pc, dc in zip(prefill_cost, decode_cost)]

def grade(sol, fx) -> dict:
    # deterministic set of requests
    requests = [
        {"prompt_len": 10, "gen_len": 5},
        {"prompt_len": 20, "gen_len": 15},
        {"prompt_len": 50, "gen_len": 2},
        {"prompt_len": 3,  "gen_len": 30},
        {"prompt_len": 25, "gen_len": 25}
    ]
    try:
        got = sol.classify_prefill_decode(requests)
    except Exception:
        return {"exact_match": 0.0}

    ref = _oracle(requests)

    ok = 1.0 if got == ref else 0.0
    return {"exact_match": ok}
