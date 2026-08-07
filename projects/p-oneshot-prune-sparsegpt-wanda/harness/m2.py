import ref
import numpy as np

def check(workdir):
    from prune.layer import prune_unstructured, correct_bias

    m = {"prune_ok": 0.0, "bias_ok": 0.0}
    w, x = ref.get_fixture()

    scores = np.random.rand(*w.shape)
    w_p, mask = prune_unstructured(w, scores, 0.5)

    expected_true = int(w.shape[1] * 0.5) * w.shape[0]
    if mask.sum() == expected_true and np.all(w_p[mask] == w[mask]) and np.all(w_p[~mask] == 0):
        m["prune_ok"] = 1.0

    bias = correct_bias(w, w_p, x)
    x_mean = np.mean(x, axis=1)
    expected_bias = (w - w_p) @ x_mean
    if bias.shape == (w.shape[0],) and np.allclose(bias, expected_bias):
        m["bias_ok"] = 1.0

    return m
