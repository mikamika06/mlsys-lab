import numpy as np


def _softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attend(q, K, V, d):
    scores = (q @ K.T) / np.sqrt(d)
    weights = _softmax_rows(scores[None, :])[0]
    return weights @ V


def _oracle(K, V, Q_obs, Q_new, budget, pool_size):
    H, n, d = K.shape
    w = Q_obs.shape[1]
    pad = pool_size // 2
    kernel = np.ones(pool_size) / pool_size

    kept_idx = []
    outputs = np.zeros((H, d), dtype=np.float64)

    for h in range(H):
        attn = _softmax_rows((Q_obs[h] @ K[h].T) / np.sqrt(d))
        raw_score = attn.sum(axis=0)

        padded = np.pad(raw_score, (pad, pad), mode="edge")
        pooled = np.convolve(padded, kernel, mode="valid")

        win = np.arange(n - w, n)
        k_extra = budget - w
        if k_extra <= 0:
            idx = np.sort(win[-budget:])
        else:
            mask = np.ones(n, dtype=bool)
            mask[win] = False
            cand = np.nonzero(mask)[0]
            top_extra = cand[np.argsort(-pooled[cand], kind="stable")[:k_extra]]
            idx = np.sort(np.concatenate([win, top_extra]))

        kept_idx.append(idx)
        outputs[h] = _attend(Q_new[h], K[h][idx], V[h][idx], d)

    return kept_idx, outputs


def _cases():
    rng = np.random.default_rng(17)
    cases = []
    for _ in range(6):
        H = int(rng.integers(1, 4))
        n = int(rng.integers(24, 48))
        d = int(rng.integers(3, 8))
        w = int(rng.integers(2, 6))
        budget = w + int(rng.integers(4, 12))
        pool_size = 3

        K = rng.standard_normal((H, n, d))
        V = rng.standard_normal((H, n, d))
        Q_obs = rng.standard_normal((H, w, d))
        Q_new = rng.standard_normal((H, d))
        cases.append((K, V, Q_obs, Q_new, budget, pool_size))
    return cases


def grade(sol, fx) -> dict:
    total_heads = 0
    correct_heads = 0
    worst_err = 0.0

    for K, V, Q_obs, Q_new, budget, pool_size in _cases():
        ref_idx, ref_out = _oracle(K, V, Q_obs, Q_new, budget, pool_size)
        H = K.shape[0]

        try:
            got = sol.snapkv_pooled_selection(K.copy(), V.copy(), Q_obs.copy(), Q_new.copy(),
                                               budget, pool_size)
            got_idx = got["kept_idx"]
            got_out = np.asarray(got["output"], dtype=np.float64)
        except Exception:
            total_heads += H
            worst_err = float("inf")
            continue

        if got_out.shape != ref_out.shape or len(got_idx) != H:
            total_heads += H
            worst_err = float("inf")
            continue

        for h in range(H):
            total_heads += 1
            try:
                gh = sorted(int(x) for x in got_idx[h])
            except Exception:
                continue
            if gh == sorted(ref_idx[h].tolist()):
                correct_heads += 1

        worst_err = max(worst_err, float(np.max(np.abs(got_out - ref_out))))

    exact_match = (correct_heads / total_heads) if total_heads else 0.0
    return {"exact_match": exact_match, "max_abs_err": worst_err}
