## Context

For a batch of raw scores $Z \in \mathbb{R}^{N \times C}$ and integer targets
$y \in \{0,\dots,C-1\}^N$, the mean softmax cross-entropy loss is

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log p_{i, y_i}, \qquad p_{ij} = \frac{e^{z_{ij}}}{\sum_k e^{z_{ik}}} .$$

Differentiating the fused softmax-then-log expression collapses to one of the tidiest results in
deep learning — the Jacobian of the softmax and the $\log$ cancel almost entirely:

$$\frac{\partial L}{\partial z_{ij}} = \frac{1}{N}\left( p_{ij} - \mathbb{1}[j = y_i] \right).$$

This is why frameworks fuse `log_softmax` and `nll_loss` into one kernel: the backward pass never
needs the full $C \times C$ softmax Jacobian, just *softmax minus one-hot*. Two properties follow
directly and are worth remembering:

* every row of the gradient sums to zero, since $\sum_j p_{ij} = 1$ and the one-hot row also sums to 1;
* the result is invariant to adding a constant to a whole row of $Z$, so a numerically stable
  implementation must subtract the row max before exponentiating. Without that shift, logits of
  magnitude a few hundred overflow `math.exp` to `inf` and the gradient becomes `nan`.

## Task

Implement the fused backward pass:

```python
def cross_entropy_backward(logits: list[list[float]], labels: list[int]) -> list[list[float]]:
    ...
```

- `logits` — list of shape $(N, C)$, unnormalised scores.
- `labels` — list of shape $(N,)$ with values in $[0, C)$.
- Returns a list of shape $(N, C)$ containing $\partial L / \partial Z$ for the **mean** reduction (i.e. including the $1/N$ factor).


Do not modify `logits` in place. The implementation must stay finite for logits of magnitude $\sim 10^3$, in both directions.

## Example

```python

logits = [[0.0, 1.0, -1.0]]
labels = [1]

g = cross_entropy_backward(logits, labels)
# softmax  -> [0.24472847, 0.66524096, 0.09003057]
# minus onehot at column 1, divided by N = 1
# g        -> [[ 0.24472847, -0.33475904,  0.09003057]]
# rows sum to 0 -> 0.0
```

## What the gate checks

The grader generates its cases with `random.Random(0)` — ordinary logits, logits scaled by $400$, a row containing a $900$ outlier, and a batch offset by $-10^3$ — and compares your output against two independent oracles it computes itself at grade time.

- `max_abs_err` $\le 10^{-9}$ — worst absolute deviation from the analytic float64 reference $(\mathrm{softmax}(Z) - \mathrm{onehot}(y))/N$, computed through a stable log-softmax. The overflow cases make an unshifted `math.exp` return `nan` and fail this outright.
- `fd_max_abs_err` $\le 10^{-5}$ — deviation from **central finite differences** $\big(L(z + h) - L(z - h)\big) / 2h$ taken on the grader's own forward loss, so the sign, the $1/N$ factor and the one-hot placement are all verified numerically rather than symbolically.
- `sum_zero_err` $\le 10^{-12}$ — the largest $|\sum_j g_{ij}|$ over all rows and cases.
