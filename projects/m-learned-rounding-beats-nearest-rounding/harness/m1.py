import ref
import numpy as np


def check(workdir):
    from quantlib.rounding import nearest_rounding, learned_rounding

    weights, scale, zero_point = ref.get_test_data()
    nn_out = nearest_rounding(weights, scale, zero_point)
    lr_out = learned_rounding(weights, scale, zero_point, steps=50)

    mse_nn = float(np.mean((weights - nn_out) ** 2))
    mse_lr = float(np.mean((weights - lr_out) ** 2))

    passed = 1.0 if mse_lr <= mse_nn else 0.0
    out = {"mse_improved": passed, "mse_nn": mse_nn, "mse_lr": mse_lr}
    if passed == 0.0:
        out["_note"] = f"Learned rounding MSE ({mse_lr}) did not beat nearest rounding MSE ({mse_nn})"
    return out
