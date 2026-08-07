import ref
import numpy as np

def check(workdir):
    from prune.eval import compare_methods

    m = {"loss_bounded_50": 0.0}
    w, x = ref.get_fixture()

    mse_mag, mse_wan = compare_methods(w, x, 0.5)
    oracle_mag, oracle_wan = ref.oracle_compare(w, x, 0.5)

    if np.isclose(mse_wan, oracle_wan, rtol=1e-4) and np.isclose(mse_mag, oracle_mag, rtol=1e-4):
        m["loss_bounded_50"] = 1.0

    return m
