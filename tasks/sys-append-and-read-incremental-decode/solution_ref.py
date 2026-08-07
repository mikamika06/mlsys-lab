import math

def incremental_decode(embeddings: list[list[float]], Wq: list[list[float]], Wk: list[list[float]], Wv: list[list[float]]) -> list[list[float]]: """ Correct implementation of incremental KV‑cache decoding.

```
Parameters
----------
embeddings : list[list[float]], shape (n, d_in)
    Token embeddings.
Wq, Wk, Wv : list[list[float]]
    Weight matrices for Q, K and V respectively.

Returns
-------
outputs : list[list[float]], shape (n, d_v)
    Incremental attention outputs.
"""
n = len(embeddings)
d_in = len(embeddings[0])
d_q = len(Wq[0])
d_k = len(Wk[0])
d_v = len(Wv[0])
sqrt_dk = math.sqrt(d_k)

cache_K_list = []
cache_V_list = []
outputs_list = []

for t in range(n):
    x_t = embeddings[t]
    Q_t = [sum(x_t[i] * Wq[i][j] for i in range(d_in)) for j in range(d_q)]
    K_t = [sum(x_t[i] * Wk[i][j] for i in range(d_in)) for j in range(d_k)]
    V_t = [sum(x_t[i] * Wv[i][j] for i in range(d_in)) for j in range(d_v)]

    cache_K_list.append(K_t)
    cache_V_list.append(V_t)

    scores = []
    for s in range(t + 1):
        dot_val = sum(Q_t[j] * cache_K_list[s][j] for j in range(d_k))
        scores.append(dot_val / sqrt_dk)

    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    sum_exp = sum(exp_scores)
    alphas = [es / sum_exp for es in exp_scores]

    out_t = [sum(alphas[s] * cache_V_list[s][v_col] for s in range(t + 1)) for v_col in range(d_v)]
    outputs_list.append(out_t)

return outputs_list
