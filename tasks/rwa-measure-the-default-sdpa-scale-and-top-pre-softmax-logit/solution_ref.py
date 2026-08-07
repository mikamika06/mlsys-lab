import math

def measure_sdpa_scale_and_top_logit(
    Q: list[list[list[list[float]]]],
    K: list[list[list[list[float]]]],
    *,
    scale: float | None = None
) -> tuple[float, float]:
    B = len(Q)
    H = len(Q[0])
    N_q = len(Q[0][0])
    d_k = len(Q[0][0][0])

    used_scale = 1.0 / math.sqrt(d_k) if scale is None else float(scale)

    top_logit = None

    for b in range(B):
        for h in range(H):
            Q_bh = Q[b][h]
            K_bh = K[b][h]
            N_k = len(K_bh)
            for i in range(N_q):
                Q_bhi = Q_bh[i]
                for j in range(N_k):
                    K_bhj = K_bh[j]
                    dot = 0.0
                    for l in range(d_k):
                        dot += Q_bhi[l] * K_bhj[l]
                    logit = dot * used_scale
                    if top_logit is None or logit > top_logit:
                        top_logit = logit

    return used_scale, float(top_logit)
