import math


def mha_forward(
    X: list[list[list[float]]],
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
    Wo: list[list[float]],
    num_heads: int,
) -> list[list[list[float]]]:
    B = len(X)
    S = len(X[0])
    E = len(X[0][0])
    d = E // num_heads

    # 1. Linear projections
    Q_proj = [[[0.0] * E for _ in range(S)] for _ in range(B)]
    K_proj = [[[0.0] * E for _ in range(S)] for _ in range(B)]
    V_proj = [[[0.0] * E for _ in range(S)] for _ in range(B)]

    for b in range(B):
        for s in range(S):
            for e in range(E):
                q_sum = 0.0
                k_sum = 0.0
                v_sum = 0.0
                for k_idx in range(E):
                    x_val = X[b][s][k_idx]
                    q_sum += x_val * Wq[k_idx][e]
                    k_sum += x_val * Wk[k_idx][e]
                    v_sum += x_val * Wv[k_idx][e]
                Q_proj[b][s][e] = q_sum
                K_proj[b][s][e] = k_sum
                V_proj[b][s][e] = v_sum

    # 2. Reshape & Transpose to (B, H, S, d)
    q = [[[[0.0] * d for _ in range(S)] for _ in range(num_heads)] for _ in range(B)]
    k = [[[[0.0] * d for _ in range(S)] for _ in range(num_heads)] for _ in range(B)]
    v = [[[[0.0] * d for _ in range(S)] for _ in range(num_heads)] for _ in range(B)]

    for b in range(B):
        for s in range(S):
            for h in range(num_heads):
                for i in range(d):
                    e_idx = h * d + i
                    q[b][h][s][i] = Q_proj[b][s][e_idx]
                    k[b][h][s][i] = K_proj[b][s][e_idx]
                    v[b][h][s][i] = V_proj[b][s][e_idx]

    # 3. Scaled dot-product attention
    scale = 1.0 / math.sqrt(d)
    out_heads = [[[[0.0] * d for _ in range(S)] for _ in range(num_heads)] for _ in range(B)]

    for b in range(B):
        for h in range(num_heads):
            for i in range(S):
                scores = [0.0] * S
                max_score = -float("inf")
                for j in range(S):
                    dot = 0.0
                    for l in range(d):
                        dot += q[b][h][i][l] * k[b][h][j][l]
                    val = dot * scale
                    scores[j] = val
                    if val > max_score:
                        max_score = val

                exp_sum = 0.0
                weights = [0.0] * S
                for j in range(S):
                    w = math.exp(scores[j] - max_score)
                    weights[j] = w
                    exp_sum += w

                for j in range(S):
                    weights[j] /= exp_sum

                for l in range(d):
                    val = 0.0
                    for j in range(S):
                        val += weights[j] * v[b][h][j][l]
                    out_heads[b][h][i][l] = val

    # 4. Transpose & Reshape back to (B, S, E)
    out_concat = [[[0.0] * E for _ in range(S)] for _ in range(B)]
    for b in range(B):
        for s in range(S):
            for h in range(num_heads):
                for i in range(d):
                    e_idx = h * d + i
                    out_concat[b][s][e_idx] = out_heads[b][h][s][i]

    # 5. Output projection Wo
    Y = [[[0.0] * E for _ in range(S)] for _ in range(B)]
    for b in range(B):
        for s in range(S):
            for e in range(E):
                sum_val = 0.0
                for k_idx in range(E):
                    sum_val += out_concat[b][s][k_idx] * Wo[k_idx][e]
                Y[b][s][e] = sum_val

    return Y
