import numpy as np


def _oracle_argmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def grade(sol, fx) -> dict:
    cases = [
        np.array(
            [
                [99, 100, 98],
                [-20, -10, -30],
                [50, 51, 49],
                [0, -100, -200],
            ],
            dtype=np.float16,
        ),
        np.array(
            [
                [300, 301, 299, 298],
                [-300, -301, -302, -303],
                [80, 81, 79, 78],
                [1, 4, 2, 3],
            ],
            dtype=np.float16,
        ),
    ]

    total = 0
    matched = 0
    for logits in cases:
        ref = _oracle_argmax(logits)
        try:
            got = np.asarray(sol.stable_softmax_argmax(logits))
        except Exception:
            return {"argmax_agreement": 0.0}

        if got.shape != ref.shape:
            return {"argmax_agreement": 0.0}

        matched += int(np.sum(got == ref))
        total += int(ref.size)

    return {"argmax_agreement": matched / total}
