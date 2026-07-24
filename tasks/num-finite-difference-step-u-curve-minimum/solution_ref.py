def optimal_fd_step(f, df, x):
    hs = [10.0 ** k for k in range(-16, 0)]
    true = df(x)
    best_h = hs[0]
    best_err = float("inf")

    for h in hs:
        estimate = (f(x + h) - f(x - h)) / (2.0 * h)
        err = abs(estimate - true)
        if err < best_err:
            best_err = err
            best_h = h

    return best_h
