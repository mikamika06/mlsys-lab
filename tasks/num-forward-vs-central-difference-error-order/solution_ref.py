import numpy as np


def finite_difference_error_orders(f, x, hs):
    hs = np.asarray(hs, dtype=np.float64)
    h0 = float(np.min(hs)) * 0.01
    derivative = (f(x + h0) - f(x - h0)) / (2.0 * h0)

    fx = f(x)
    forward_errors = np.empty_like(hs, dtype=np.float64)
    central_errors = np.empty_like(hs, dtype=np.float64)

    for i, h in enumerate(hs):
        forward = (f(x + h) - fx) / h
        central = (f(x + h) - f(x - h)) / (2.0 * h)
        forward_errors[i] = abs(forward - derivative)
        central_errors[i] = abs(central - derivative)

    return forward_errors, central_errors
