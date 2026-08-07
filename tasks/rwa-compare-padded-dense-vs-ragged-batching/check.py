import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _padded_dense_oracle(sequences):
    lengths = [len(x) for x in sequences]
    n = len(sequences)
    d = len(sequences[0][0])
    m = max(lengths)
    padded = np.zeros((n, m, d), dtype=np.float64)
    mask = np.zeros((n, m), dtype=bool)
    for i, x in enumerate(sequences):
        l = len(x)
        padded[i, :l] = x
        mask[i, :l] = True

    outputs = []
    for b in range(n):
        q = padded[b]
        scores = (q @ q.T) / np.sqrt(d)
        scores[:, ~mask[b]] = -np.inf
        probs = _softmax(scores)
        outputs.append((probs @ padded[b])[mask[b]])
    return outputs


def _ragged_pair_count(sequences):
    return sum(len(x) ** 2 for x in sequences)


def grade(sol, fx) -> dict:
    cases = [
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [0.5, -1.0], [1.0, 3.0]],
        ],
        [
            [[1.0, 2.0, 3.0]],
            [[0.5, 0.5, 0.5], [1.0, -1.0, 2.0]],
            [[2.0, 0.0, 1.0], [3.0, 1.0, -1.0], [0.0, 2.0, 2.0], [1.0, 1.0, 0.0]],
        ],
    ]

    max_err = 0.0
    ratio_ok = 1.0

    for seqs in cases:
        ref = _padded_dense_oracle(seqs)
        try:
            got, ratio = sol.ragged_attention_compare(seqs)
        except Exception:
            return {"max_abs_err": float("inf"), "size_ratio": 0.0}

        if len(got) != len(ref):
            max_err = float("inf")
            ratio_ok = 0.0
            break

        for a, b in zip(got, ref):
            if np.asarray(a).shape != b.shape:
                max_err = float("inf")
                break
            max_err = max(max_err, float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b))))

        lengths = [len(x) for x in seqs]
        expected_ratio = (len(seqs) * max(lengths) ** 2) / _ragged_pair_count(seqs)
        if ratio != expected_ratio:
            ratio_ok = 0.0

    return {
        "max_abs_err": max_err,
        "size_ratio": ratio_ok,
    }
