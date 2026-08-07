import math


def _e4m3_roundtrip_row(row, scale):
    out = []
    for val in row:
        y = val / scale
        if y > 448.0:
            y = 448.0
        elif y < -448.0:
            y = -448.0
        sign = -1.0 if y < 0.0 else (1.0 if y > 0.0 else 0.0)
        ay = abs(y)
        if ay == 0.0:
            val_out = 0.0
        elif ay < 2 ** -6:
            val_out = round(ay / (2 ** -9)) * (2 ** -9)
        else:
            exp = math.floor(math.log2(max(ay, 2 ** -9)))
            if exp < -6:
                exp = -6
            elif exp > 7:
                exp = 7
            frac = ay / (2.0 ** exp) - 1.0
            mant = round(frac * 8.0) / 8.0
            val_out = (1.0 + mant) * (2.0 ** exp)
        out.append(sign * val_out * scale)
    return out


def fp8_attention_output(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    max_k = 0.0
    for row in K:
        for val in row:
            if abs(val) > max_k:
                max_k = abs(val)
    sk = max(max_k / 448.0, 1e-12)

    max_v = 0.0
    for row in V:
        for val in row:
            if abs(val) > max_v:
                max_v = abs(val)
    sv = max(max_v / 448.0, 1e-12)

    K_hat = [_e4m3_roundtrip_row(row, sk) for row in K]
    V_hat = [_e4m3_roundtrip_row(row, sv) for row in V]

    d = len(Q[0])
    scale_factor = math.sqrt(d)

    scores = []
    for q_row in Q:
        score_row = []
        for k_row in K_hat:
            dot = sum(a * b for a, b in zip(q_row, k_row))
            score_row.append(dot / scale_factor)
        scores.append(score_row)

    probs = []
    for row in scores:
        max_val = max(row)
        exp_row = [math.exp(val - max_val) for val in row]
        sum_exp = sum(exp_row)
        probs.append([val / sum_exp for val in exp_row])

    m = len(Q)
    v_rows = len(V_hat)
    v_cols = len(V_hat[0])
    out = []
    for i in range(m):
        out_row = []
        for j in range(v_cols):
            val = sum(probs[i][k] * V_hat[k][j] for k in range(v_rows))
            out_row.append(val)
        out.append(out_row)

    return out
