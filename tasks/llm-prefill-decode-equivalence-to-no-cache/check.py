import numpy as np
import random

d = 16

# Generate identical weights using Python's random module (matching solution_ref.py)
random.seed(0)
ref_W_ih = [[random.gauss(0, 1) for _ in range(d)] for _ in range(d)]
ref_W_hh = [[random.gauss(0, 1) for _ in range(d)] for _ in range(d)]
ref_b = [random.gauss(0, 1) for _ in range(d)]

W_ih = np.array(ref_W_ih, dtype=np.float64)
W_hh = np.array(ref_W_hh, dtype=np.float64)
b = np.array(ref_b, dtype=np.float64)

def _reference(inputs):
    seq_len = len(inputs)
    # no‑cache strategy
    no_cache = []
    for L in range(1, seq_len + 1):
        h_prev = np.zeros(d, dtype=np.float64)
        for i in range(L):
            h_prev = np.tanh(W_ih @ np.array(inputs[i], dtype=np.float64) + W_hh @ h_prev + b)
        no_cache.append(h_prev)
    no_cache = [row.tolist() for row in no_cache]

    # cache strategy
    cache = []
    h_prev = np.zeros(d, dtype=np.float64)
    for i in range(seq_len):
        h_prev = np.tanh(W_ih @ np.array(inputs[i], dtype=np.float64) + W_hh @ h_prev + b)
        cache.append(h_prev)
    cache = [row.tolist() for row in cache]

    return no_cache, cache

def grade(sol, fx) -> dict:
    # generate a random test sequence
    rng = np.random.RandomState(1)
    seq_len = 10
    inputs = rng.randn(seq_len, d).astype(np.float64).tolist()

    try:
        got_no, got_cache = sol.prefill_decode_equiv(inputs)
    except Exception as e:
        return {"max_abs_err": float("inf")}

    ref_no, ref_cache = _reference(inputs)

    # compute maximum absolute error
    err1 = np.max(np.abs(np.array(got_no) - np.array(ref_no)))
    err2 = np.max(np.abs(np.array(got_cache) - np.array(ref_cache)))
    max_err = max(err1, err2)
    return {"max_abs_err": float(max_err)}
