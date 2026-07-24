import numpy as np


def _oracle(layer_gates, head_gates, dim_gates, target):
    target_L, target_H, target_d_ff = target

    def top_indices(values, count):
        values = np.asarray(values)
        order = np.lexsort((np.arange(values.shape[0]), -values))
        return np.sort(order[:count]).tolist()

    layers = top_indices(layer_gates, target_L)
    heads = []
    for layer in layers:
        heads.append(top_indices(head_gates[layer], target_H))
    dims = top_indices(dim_gates, target_d_ff)

    return {
        "layers": layers,
        "heads": heads,
        "dims": dims,
    }


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.2, 0.9, 0.5]),
            np.array([[0.1, 0.8], [0.7, 0.3], [0.9, 0.4]]),
            np.array([0.5, 0.1, 0.8]),
            (2, 1, 2),
        ),
        (
            np.array([0.4, 0.4, 0.9, 0.1]),
            np.array([
                [0.5, 0.2, 0.8],
                [0.6, 0.7, 0.1],
                [0.3, 0.9, 0.4],
                [0.8, 0.2, 0.5],
            ]),
            np.array([0.2, 0.2, 0.7, 0.6, 0.1]),
            (3, 2, 3),
        ),
        (
            np.array([0.8, 0.2, 0.6]),
            np.array([
                [0.9, 0.9, 0.1],
                [0.4, 0.3, 0.2],
                [0.5, 0.5, 0.5],
            ]),
            np.array([0.1, 0.9, 0.4, 0.9]),
            (2, 2, 2),
        ),
    ]

    ok = 1.0
    for layer_gates, head_gates, dim_gates, target in cases:
        expected = _oracle(layer_gates, head_gates, dim_gates, target)
        try:
            got = sol.map_mask_to_arch(
                layer_gates.copy(),
                head_gates.copy(),
                dim_gates.copy(),
                target,
            )
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
