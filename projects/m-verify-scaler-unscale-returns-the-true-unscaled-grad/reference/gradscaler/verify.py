import numpy as np


def verify_unscaled_grad(scaler, optimizer, expected_true_grads):
    unscaled_grads = scaler.unscale_(optimizer)
    max_err = 0.0
    for g_group, e_group in zip(unscaled_grads, expected_true_grads):
        for g, e in zip(g_group, e_group):
            if g is not None and e is not None:
                err = float(np.max(np.abs(g - e)))
                if err > max_err:
                    max_err = err
    return max_err
