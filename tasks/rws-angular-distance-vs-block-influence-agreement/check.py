import numpy as np


def _rank_desc(values):
    return list(np.argsort(-np.asarray(values), kind="stable"))


def _rank_positions(order):
    pos = np.empty(len(order), dtype=np.float64)
    for i, idx in enumerate(order):
        pos[idx] = i + 1
    return pos


def _spearman(order_a, order_b):
    ra = _rank_positions(order_a)
    rb = _rank_positions(order_b)
    n = len(ra)
    return float(1.0 - 6.0 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def _ref(states):
    angles = []
    influences = []
    eps = 1e-12
    for layer in states:
        before = np.asarray(layer["before"], dtype=np.float64)
        after = np.asarray(layer["after"], dtype=np.float64)
        denom = np.linalg.norm(before) * np.linalg.norm(after) + eps
        cosine = float(np.dot(before, after) / denom)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        angles.append(float(np.arccos(cosine)))
        influences.append(float(np.linalg.norm(after - before) / (np.linalg.norm(before) + eps)))
    angle_order = _rank_desc(angles)
    influence_order = _rank_desc(influences)
    return angle_order, influence_order, _spearman(angle_order, influence_order)


def grade(sol, fx) -> dict:
    cases = [
        [
            {"before": np.array([1.0, 0.0]), "after": np.array([0.8, 0.6])},
            {"before": np.array([1.0, 1.0]), "after": np.array([1.2, 1.1])},
            {"before": np.array([0.0, 2.0]), "after": np.array([1.0, 1.5])},
            {"before": np.array([2.0, 2.0]), "after": np.array([2.0, 2.1])},
        ],
        [
            {"before": np.array([3.0, -1.0, 0.5]), "after": np.array([2.5, -0.2, 0.7])},
            {"before": np.array([1.0, 2.0, 3.0]), "after": np.array([0.8, 2.1, 2.9])},
            {"before": np.array([-1.0, 0.0, 1.0]), "after": np.array([0.0, 1.0, 1.0])},
        ],
    ]
    ok = 1.0
    for states in cases:
        ref = _ref(states)
        try:
            got = sol.angular_distance_vs_block_influence(states)
            angle_order, influence_order, rho = got
            if list(angle_order) != ref[0]:
                ok = 0.0
                break
            if list(influence_order) != ref[1]:
                ok = 0.0
                break
            if abs(float(rho) - ref[2]) > 1e-6:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
