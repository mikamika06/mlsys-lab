import ref
import numpy as np

def check(workdir):
    from prune.importance import score_magnitude, score_wanda

    m = {"mag_ok": 0.0, "wanda_ok": 0.0}
    w, x = ref.get_fixture()

    sm = score_magnitude(w)
    sw = score_wanda(w, x)

    if sm.shape == w.shape and np.allclose(sm, np.abs(w)):
        m["mag_ok"] = 1.0

    x_norm = np.linalg.norm(x, axis=1)
    expected_sw = np.abs(w) * x_norm[None, :]
    if sw.shape == w.shape and np.allclose(sw, expected_sw):
        m["wanda_ok"] = 1.0

    return m
