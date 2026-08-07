def mixed_rank_sgmv(x, adapter_ids, adapters):
    out = []
    for i in range(len(x)):
        xi = x[i]
        a, b = adapters[int(adapter_ids[i])]

        # Matrix-vector multiplication: A_i * x_i
        rk = len(a)
        d = len(xi)
        ax = [0.0] * rk
        for r in range(rk):
            row_sum = 0.0
            for c in range(d):
                row_sum += a[r][c] * xi[c]
            ax[r] = row_sum

        # Matrix-vector multiplication: B_i * (A_i * x_i)
        b_res = [0.0] * d
        for r in range(d):
            row_sum = 0.0
            for c in range(rk):
                row_sum += b[r][c] * ax[c]
            b_res[r] = row_sum

        # y_i = x_i + B_i(A_i x_i)
        yi = [xi[c] + b_res[c] for c in range(d)]
        out.append(yi)

    return out
