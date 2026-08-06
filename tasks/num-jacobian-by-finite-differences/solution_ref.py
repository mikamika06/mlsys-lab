def jacobian_fd(f, x, eps=1e-6):
    x_list = [float(v) for v in x]
    y_list = [float(v) for v in f(x_list)]
    m = len(y_list)
    n = len(x_list)
    J = [[0.0 for _ in range(n)] for _ in range(m)]

    for j in range(n):
        xp = list(x_list)
        xm = list(x_list)
        xp[j] += eps
        xm[j] -= eps
        yp = [float(v) for v in f(xp)]
        ym = [float(v) for v in f(xm)]
        for i in range(m):
            J[i][j] = (yp[i] - ym[i]) / (2.0 * eps)

    return J
