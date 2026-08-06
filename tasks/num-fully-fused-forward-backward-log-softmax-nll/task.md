## Context

Training a classifier with logits $Z \in \mathbb{R}^{N \times C}$ and integer
targets $t \in \{0, \dots, C-1\}^{N}$ needs two things every step: the
negative-log-likelihood (NLL) loss and its gradient with respect to $Z$. Doing
this as separate library calls — `log_softmax` then `nll_loss`, then
autodiff for the backward pass — recomputes the same log-sum-exp twice and
materializes an extra $N \times C$ array of log-probabilities that a fused
kernel does not need.

The stable log-softmax of row $i$ is

$$
\log \operatorname{softmax}(Z_i)_c = Z_{i,c} - \operatorname{LSE}(Z_i), \qquad
\operatorname{LSE}(Z_i) = m_i + \log \sum_{c} e^{Z_{i,c} - m_i}, \qquad
m_i = \max_c Z_{i,c} .
$$

The mean NLL loss over the batch is

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \operatorname{softmax}(Z_i)_{t_i} .
$$

Differentiating $\mathcal{L}$ with respect to the logits gives a closed form
that only needs the softmax probabilities $P_{i,c} = e^{\log\operatorname{softmax}(Z_i)_c}$:

$$
\frac{\partial \mathcal{L}}{\partial Z_{i,c}} = \frac{1}{N}\Bigl(P_{i,c} - \mathbb{1}[c = t_i]\Bigr) .
$$

Fusing forward and backward means computing $\operatorname{LSE}$ once, reusing
it for both the loss and $P$, and never forming a separate autodiff graph.

## Task

Implement `fused_log_softmax_nll(logits, targets)`:

```python
def fused_log_softmax_nll(logits: list[list[float]], targets: list[int]):
    ...
```

* `logits` is a 2-D `float64` array of shape $(N, C)$.
* `targets` is a 1-D integer array of shape $(N,)$ with values in $[0, C)$.

Return a tuple `(loss, dlogits)` where `loss` is a Python `float` equal to
$\mathcal{L}$ above, and `dlogits` is a `float64` list of shape
$(N, C)$ equal to $\partial \mathcal{L} / \partial Z$. Both quantities must be
computed with a numerically stable log-sum-exp (subtract the row max before
exponentiating) so the function stays accurate for logits with large
magnitude.

## Example

```python
logits = [[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]]
targets = [0, 2]
loss, dlogits = fused_log_softmax_nll(logits, targets)
# loss  ~= 0.4170   (mean NLL over the 2 rows)
# dlogits ~= softmax(logits) with 1/N subtracted at the target column of each row
```

## What the gate checks

The grader builds several random `(logits, targets)` batches — including ones
scaled to large magnitude, where a naive `exp` without max-subtraction would
overflow — and compares your `loss` and `dlogits` against an independently
computed reference using the same stable formulas. Two gates:
$\mathrm{max\_abs\_err\_loss} \le 10^{-9}$ on the scalar loss, and
$\mathrm{max\_abs\_err\_grad} \le 10^{-9}$ on the elementwise gradient error
$\max_{i,c} |\hat{Z}'_{i,c} - Z'_{i,c}|$. Both must pass.
