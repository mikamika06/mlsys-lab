import math

def grade(sol, fx=None):
    def rope_ref(x, pos):
        B = len(x)
        S = len(x[0])
        H = len(x[0][0])
        D = len(x[0][0][0])
        half_D = D // 2
        freqs = [1.0 / (10000.0 ** ((2 * i) / D)) for i in range(half_D)]

        out = []
        for b in range(B):
            batch_out = []
            for s in range(S):
                m = float(pos[s])
                seq_out = []
                for h in range(H):
                    head_data = x[b][s][h]
                    new_head_data = [0.0] * D
                    for i in range(half_D):
                        angle = m * freqs[i]
                        c = math.cos(angle)
                        sin_val = math.sin(angle)
                        x_even = head_data[2 * i]
                        x_odd = head_data[2 * i + 1]
                        new_head_data[2 * i] = x_even * c - x_odd * sin_val
                        new_head_data[2 * i + 1] = x_odd * c + x_even * sin_val
                    seq_out.append(new_head_data)
                batch_out.append(seq_out)
            out.append(batch_out)
        return out

    shapes_data = [
        (2, 4, 2, 4),
        (1, 3, 2, 6),
    ]

    max_err = 0.0
    for B, S, H, D in shapes_data:
        x = [[[[float(i + b + s + h) for i in range(D)] for h in range(H)] for s in range(S)] for b in range(B)]
        pos = list(range(S))
        try:
            out_student = sol.apply_rope(x, pos)
        except Exception:
            max_err = 1.0
            break
        out_ref = rope_ref(x, pos)

        for b in range(B):
            for s in range(S):
                for h in range(H):
                    for d in range(D):
                        err = abs(out_student[b][s][h][d] - out_ref[b][s][h][d])
                        if err > max_err:
                            max_err = err

    return {
        "max_abs_err": float(max_err),
        "line_count": 10
    }
