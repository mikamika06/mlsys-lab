import numpy as np

def _reference(logits, bias):
    seq_len = logits.shape[0]
    mask = np.triu(np.full_like(logits, fill_value=-np.inf), k=1)
    masked = logits + bias + mask
    max_vals = np.max(masked, axis=-1, keepdims=True)
    exp_shift = np.exp(masked - max_vals)
    probs = exp_shift / np.sum(exp_shift, axis=-1, keepdims=True)
    return probs

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for seq_len in [2, 3, 5]:
        logits_arr = rng.standard_normal((seq_len, seq_len))
        bias_arr = rng.uniform(-1, 1, (seq_len, seq_len))

        logits_list = logits_arr.tolist()
        bias_list = bias_arr.tolist()

        try:
            cand = sol.causal_alibi_logits(logits_list, bias_list)
        except Exception:
            return {"max_abs_err": float("inf")}

        cand_arr = np.array(cand, dtype=np.float64)
        ref = _reference(logits_arr, bias_arr)
        err = np.max(np.abs(cand_arr - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
