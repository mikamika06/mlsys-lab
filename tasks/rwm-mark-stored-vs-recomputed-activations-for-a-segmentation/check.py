import numpy as np

def _ref(num_layers, seg_lengths):
    if sum(seg_lengths) != num_layers:
        raise ValueError("segment lengths must sum to num_layers")
    boundaries = np.cumsum([0] + list(seg_lengths))[:-1]
    labels = np.zeros(num_layers, dtype=int)
    labels[boundaries] = 1
    return labels

def grade(sol, fx) -> dict:
    ok = 1.0
    for _ in range(10):
        num_layers = np.random.randint(5, 50)
        k = np.random.randint(2, min(10, num_layers))
        # generate random segment lengths that sum to num_layers
        cuts = sorted(np.random.choice(range(1, num_layers), k-1, replace=False))
        seg_lengths = [cuts[0]] + \
                      [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + \
                      [num_layers - cuts[-1]]
        try:
            got = sol.mark_activations(num_layers, seg_lengths)
            ref = _ref(num_layers, seg_lengths)
        except Exception:
            ok = 0.0
            break
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
