## Context

Cross‑entropy is a standard loss for classification problems. For a single token with true class $t$ and logits $\mathbf{z}\in\mathbb R^{V}$ the per‑token loss is
$$
\ell(\mathbf{z}, t) = -\log \frac{\exp(z_t)}{\sum_{v=1}^{V} \exp(z_v)}
  = -\,\bigl( z_t - \log \sum_{v}\exp(z_v)\bigr).
$$

When a batch of sequences is processed we usually average the loss over all *valid* tokens.  
If $M$ is an optional mask with entries $m_{ij}\in\{0,1\}$ indicating whether token $(i,j)$ should contribute, the reduction is
$$
L = \frac{\sum_{i=1}^{B}\;\sum_{j=1}^{T} m_{ij}\,\ell(\mathbf{z}_{ij}, t_{ij})}
         {\sum_{i=1}^{B}\;\sum_{j=1}^{T} m_{ij}} .
$$
When $M$ is omitted the denominator becomes $BT$, i.e. a mean over all tokens.

## Task

Implement `cross_entropy_loss(logits, targets, mask=None)` that returns a 1‑D list of shape `(batch_size,)` containing the loss for each sequence.  
The function must:

* work with logits of shape `(B, T, V)`, integer targets of shape `(B, T)` and an optional boolean mask of shape `(B, T)`;
* compute the per‑token cross‑entropy using a numerically stable log‑softmax;
* average over the valid tokens as described above.

The result should be `float32` or `float64`; any precision that passes the gate is acceptable.

## Example

```python
logits = [[[2.0, 0.5], [1.0, 3.0]],
                   [[0.0, 1.0], [4.0, -1.0]]]
targets = [[0, 1],
                    [1, 0]]
mask    = [[True, False],
                    [True, True]]

loss = cross_entropy_loss(logits, targets, mask)
print(loss)   # e.g. array([0.693..., 0.223...], dtype=float32)
```

## What the gate checks

The grader computes a reference implementation with Python and compares your output using
`arena.scorers.max_abs_err`.  
Your solution must satisfy `max_abs_err ≤ 1e-6`.
