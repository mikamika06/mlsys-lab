import math

def decoupled_rope_score(q_lat, k_lat, q_rope, k_rope):
    """Concatenate latent and rope-head, then scaled dot-product + softmax."""
    B = len(q_lat)
    H = len(q_lat[0])
    N = len(q_lat[0][0])
    D_l = len(q_lat[0][0][0])
    D_r = len(q_rope[0][0][0])
    D = D_l + D_r

    Q = []
    for b in range(B):
        batch_Q = []
        for h in range(H):
            head_Q = []
            for i in range(N):
                row_Q = q_lat[b][h][i] + q_rope[b][h][i]
                head_Q.append(row_Q)
            batch_Q.append(head_Q)
        Q.append(batch_Q)

    K = []
    for b in range(B):
        batch_K = []
        for h in range(H):
            head_K = []
            for i in range(N):
                row_K = k_lat[b][h][i] + k_rope[b][h][i]
                head_K.append(row_K)
            batch_K.append(head_K)
        K.append(batch_K)

    scale = 1.0 / math.sqrt(D)

    out = []
    for b in range(B):
        batch_out = []
        for h in range(H):
            head_out = []
            for i in range(N):
                row_out = []
                for j in range(N):
                    dot = 0.0
                    for d in range(D):
                        dot += Q[b][h][i][d] * K[b][h][j][d]
                    row_out.append(dot * scale)
                head_out.append(row_out)
            batch_out.append(head_out)
        out.append(batch_out)

    for b in range(B):
        for h in range(H):
            for i in range(N):
                max_val = out[b][h][i][0]
                for j in range(1, N):
                    if out[b][h][i][j] > max_val:
                        max_val = out[b][h][i][j]

                sum_exp = 0.0
                row_vals = []
                for j in range(N):
                    val = math.exp(out[b][h][i][j] - max_val)
                    row_vals.append(val)
                    sum_exp += val

                for j in range(N):
                    row_vals[j] /= sum_exp
                out[b][h][i] = row_vals

    return out
