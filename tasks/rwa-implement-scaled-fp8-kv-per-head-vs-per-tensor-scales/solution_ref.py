import math


def _e4m3(val):
    ax = abs(val)
    if ax == 0.0:
        return 0.0
    vals = min(ax, 448.0)
    exp = max(math.floor(math.log2(vals)), -6)
    base = math.pow(2.0, exp)
    mant = vals / base - 1.0
    mant_q = round(mant * 8.0) / 8.0
    vals_q = min(base * (1.0 + mant_q), 448.0)
    sign = 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)
    return sign * vals_q


def _qd(x, per_head):
    H = len(x)
    N = len(x[0])
    D = len(x[0][0])

    if per_head:
        scales = []
        for i in range(H):
            max_val = 0.0
            for j in range(N):
                for k in range(D):
                    val = abs(x[i][j][k])
                    if val > max_val:
                        max_val = val
            s = max_val / 448.0
            if s < 1e-12:
                s = 1e-12
            scales.append(s)
    else:
        max_val = 0.0
        for i in range(H):
            for j in range(N):
                for k in range(D):
                    val = abs(x[i][j][k])
                    if val > max_val:
                        max_val = val
        scale = max_val / 448.0
        if scale < 1e-12:
            scale = 1e-12
        scales = [scale] * H

    out = []
    for i in range(H):
        s = scales[i]
        h_matrix = []
        for j in range(N):
            row = []
            for k in range(D):
                scaled_val = x[i][j][k] / s
                q_val = _e4m3(scaled_val)
                row.append(q_val * s)
            h_matrix.append(row)
        out.append(h_matrix)
    return out


def scaled_fp8_kv_attention(K: list[list[list[float]]], V: list[list[list[float]]], Q: list[list[list[float]]], per_head: bool) -> list[list[list[float]]]:
    Kd = _qd(K, per_head)
    Vd = _qd(V, per_head)

    H = len(Q)
    M = len(Q[0])
    D = len(Q[0][0])
    N = len(Kd[0])
    V_D = len(Vd[0][0])

    scale_factor = math.sqrt(D)

    out = []
    for h in range(H):
        q_h = Q[h]
        kd_h = Kd[h]
        vd_h = Vd[h]

        logits = []
        for i in range(M):
            row = []
            for j in range(N):
                acc = 0.0
                for d in range(D):
                    acc += q_h[i][d] * kd_h[j][d]
                row.append(acc / scale_factor)
            logits.append(row)

        probs = []
        for i in range(M):
            row = logits[i]
            row_max = max(row)
            exp_row = [math.exp(val - row_max) for val in row]
            sum_exp = sum(exp_row)
            probs.append([val / sum_exp for val in exp_row])

        head_out = []
        for i in range(M):
            out_row = []
            for vd_col in range(V_D):
                acc = 0.0
                for n in range(N):
                    acc += probs[i][n] * vd_h[n][vd_col]
                out_row.append(acc)
            head_out.append(out_row)
        out.append(head_out)

    return out
