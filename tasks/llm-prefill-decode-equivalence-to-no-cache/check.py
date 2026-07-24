import numpy as np

# deterministic weights used by both reference and candidate implementations
np.random.seed(0)
d = 16
W_ih = np.random.randn(d, d).astype(np.float64)
W_hh = np.random.randn(d, d).astype(np.float64)
b = np.random.randn(d).astype(np.float64)

def _reference(inputs: np.ndarray):
    seq_len = inputs.shape[0]
    # no‑cache strategy
    no_cache = []
    for L in range(1, seq_len + 1):
        h_prev = np.zeros(d, dtype=np.float64)
        for i in range(L):
            h_prev = np.tanh(W_ih @ inputs[i] + W_hh @ h_prev + b)
        no_cache.append(h_prev)
    no_cache = np.stack(no_cache)

    # cache strategy
    cache = []
    h_prev = np.zeros(d, dtype=np.float64)
    for i in range(seq_len):
        h_prev = np.tanh(W_ih @ inputs[i] + W_hh @ h_prev + b)
        cache.append(h_prev)
    cache = np.stack(cache)

    return no_cache, cache

def grade(sol, fx) -> dict:
    # generate a random test sequence
    rng = np.random.RandomState(1)
    seq_len = 10
    inputs = rng.randn(seq_len, d).astype(np.float64)

    try:
        got_no, got_cache = sol.prefill_decode_equiv(inputs)
    except Exception as e:
        return {"max_abs_err": float("inf")}

    ref_no, ref_cache = _reference(inputs)

    # compute maximum absolute error
    err1 = np.max(np.abs(got_no - ref_no))
    err2 = np.max(np.abs(got_cache - ref_cache))
    max_err = max(err1, err2)
    return {"max_abs_err": float(max_err)}
