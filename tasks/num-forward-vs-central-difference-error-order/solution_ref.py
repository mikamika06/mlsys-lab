import numpy as np


def finite_difference_error_orders(f, x, hs):
    hs = np.asarray(hs, dtype=np.float64)
    
    min_h = hs[0]
    for h in hs:
        if h < min_h:
            min_h = h
    h0 = float(min_h) * 0.01
    
    derivative = (f(x + h0) - f(x - h0)) / (2.0 * h0)

    fx = f(x)
    forward_errors_list = []
    central_errors_list = []

    for h in hs:
        forward = (f(x + h) - fx) / h
        central = (f(x + h) - f(x - h)) / (2.0 * h)
        forward_errors_list.append(abs(forward - derivative))
        central_errors_list.append(abs(central - derivative))

    forward_errors = np.asarray(forward_errors_list, dtype=np.float64)
    central_errors = np.asarray(central_errors_list, dtype=np.float64)

    return forward_errors, central_errors
