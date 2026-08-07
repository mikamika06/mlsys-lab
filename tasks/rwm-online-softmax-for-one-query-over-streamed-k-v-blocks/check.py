import math


def ref_attention(q: list[float], K_blocks: list[list[list[float]]], V_blocks: list[list[list[float]]]) -> list[float]:
    d = len(q)
    scale = math.sqrt(float(d))

    all_k = []
    all_v = []
    for Kb, Vb in zip(K_blocks, V_blocks):
        for k_row in Kb:
            all_k.append(k_row)
        for v_row in Vb:
            all_v.append(v_row)

    if not all_k:
        return []

    dv = len(all_v[0])
    scores = []
    for k_row in all_k:
        dot = sum(q[j] * k_row[j] for j in range(d))
        scores.append(dot / scale)

    max_s = max(scores)
    weights = [math.exp(s - max_s) for s in scores]
    sum_w = sum(weights)

    out = [0.0] * dv
    for c in range(dv):
        out[c] = sum(weights[i] * all_v[i][c] for i in range(len(all_k))) / sum_w
    return out


def grade(sol, fx=None) -> dict:
    test_cases = [
        (
            [1.0, 0.0],
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0]],
            ],
            [
                [[10.0], [20.0]],
                [[30.0]],
            ],
        ),
        (
            [2.0, -1.0, 0.5],
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ],
        ),
        (
            [0.5, -0.5, 1.5, 2.0],
            [
                [[1.0, 2.0, -1.0, 0.0], [0.0, 1.0, 1.0, -2.0]],
                [[-1.0, 0.0, 2.0, 1.0]],
                [[0.5, -1.0, 0.0, 0.5], [1.5, 0.5, -0.5, 1.0]],
            ],
            [
                [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]],
                [[2.0, 2.0, 0.0]],
                [[-1.0, 0.5, 1.5], [1.0, 1.0, 1.0]],
            ],
        ),
        (
            [1.0, 2.0],
            [
                [[0.5, 0.5]],
                [[1.0, -1.0]],
                [[-0.5, 2.0]],
            ],
            [
                [[5.0]],
                [[15.0]],
                [[25.0]],
            ],
        ),
    ]

    max_abs_err = 0.0

    for q, K_blocks, V_blocks in test_cases:
        expected = ref_attention(q, K_blocks, V_blocks)
        try:
            actual = sol.online_attention(q, K_blocks, V_blocks)
        except Exception:
            return {"pass": False, "max_abs_err": float('inf')}

        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return {"pass": False, "max_abs_err": float('inf')}

        for a, e in zip(actual, expected):
            try:
                err = abs(float(a) - float(e))
                if math.isnan(err):
                    return {"pass": False, "max_abs_err": float('inf')}
                if err > max_abs_err:
                    max_abs_err = err
            except (ValueError, TypeError):
                return {"pass": False, "max_abs_err": float('inf')}

    return {
        "pass": max_abs_err <= 1e-6,
        "max_abs_err": max_abs_err,
    }
