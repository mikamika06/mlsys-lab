import numpy as np


def _oracle(token_budget, num_running, prefill_remaining):
    decode_tokens = min(num_running, token_budget)
    prefill_chunk = min(prefill_remaining, token_budget - decode_tokens)
    return decode_tokens, prefill_chunk


def grade(sol, fx) -> dict:
    state = np.asarray(fx["state"], dtype=np.int64)

    ok = 1.0
    for row in state:
        token_budget, num_running, prefill_remaining = (int(x) for x in row)
        expected = _oracle(token_budget, num_running, prefill_remaining)
        try:
            got = sol.pack_step(token_budget, num_running, prefill_remaining)
            got = (int(got[0]), int(got[1]))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
