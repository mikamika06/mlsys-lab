## Context

Structured 2:4 sparsity keeps exactly two weights out of every group of four.
A mask $M \in \{0,1\}^4$ selects the retained weights in each group.

A one-shot magnitude pruning method selects the two largest absolute weights:

$$
M_{\mathrm{mag}}(w)_i =
\begin{cases}
1 & \text{if } |w_i| \text{ is among the two largest values in the group}\\
0 & \text{otherwise.}
\end{cases}
$$

A learned soft mask uses converged importance logits $z$ instead. The two
largest logits determine the retained positions:

$$
M_{\mathrm{soft}}(z)_i =
\begin{cases}
1 & \text{if } z_i \text{ is among the two largest values in the group}\\
0 & \text{otherwise.}
\end{cases}
$$

The masked reconstruction error for one weight group is:

$$
E(w,M)=\lVert w-w\odot M\rVert^2
=
\sum_i (w_i-w_iM_i)^2 .
$$

The goal is to compare the learned mask against the traditional magnitude
mask using the same weight groups.

## Task

Implement `compare_2_4_masks(weights, logits)`.

Arguments:
- `weights`: a list of shape $(n,4)$ containing weight groups.
- `logits`: a list of shape $(n,4)$ containing converged learned-mask
  logits.

Return a dictionary with exactly these keys:

- `"soft_retained"`: retained absolute-weight sums for the learned-logit mask.
- `"magnitude_retained"`: retained absolute-weight sums for the magnitude mask.
- `"soft_error"`: per-group reconstruction errors for the learned-logit mask.
- `"magnitude_error"`: per-group reconstruction errors for the magnitude mask.
- `"better"`: `"soft"`, `"magnitude"`, or `"tie"` depending on which method
  has the lower total reconstruction error.

Use stable descending ordering for ties in top-2 selection.

## Example

```python

weights = [[3.0, -1.0, 2.0, 0.5]]
logits = [[0.1, 5.0, 4.0, 3.0]]

result = compare_2_4_masks(weights, logits)

# The learned mask keeps -1.0 and 2.0.
# The magnitude mask keeps 3.0 and 2.0.
```

## What the gate checks

The grader recomputes both masks from a Python oracle and compares every numeric
output. The `mse` value is the mean squared difference between the returned
numeric arrays and the oracle arrays. The `better_match` value checks that the
returned winner agrees with the oracle comparison of total errors.

A correct implementation must match the oracle outputs exactly.
