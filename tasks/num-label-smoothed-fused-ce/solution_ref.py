import math

def label_smoothed_fused_ce(logits: list[list[float]], targets: list[int], eps: float = 0.1) -> float: """ Numerically stable label-smoothed cross-entropy.

```
Parameters
----------
logits  : list of list of float, shape (N, K) — unnormalized scores
targets : list of int, shape (N,)            — integer class indices
eps     : float                              — smoothing factor in [0, 1]

Returns
-------
float — mean cross-entropy over the batch
"""
N = len(logits)
K = len(logits[0])

q_smooth = [[eps / K for _ in range(K)] for _ in range(N)]
for i in range(N):
    q_smooth[i][targets[i]] += 1.0 - eps

losses = [0.0] * N

for i in range(N):
    max_val = logits[i][0]
    for j in range(1, K):
        if logits[i][j] > max_val:
            max_val = logits[i][j]

    sum_exp = 0.0
    for j in range(K):
        sum_exp += math.exp(logits[i][j] - max_val)
    log_Z = math.log(sum_exp)

    loss_i = 0.0
    for j in range(K):
        val = logits[i][j] - max_val - log_Z
        loss_i -= q_smooth[i][j] * val
    losses[i] = loss_i

total_loss = 0.0
for i in range(N):
    total_loss += losses[i]
return float(total_loss / N)
