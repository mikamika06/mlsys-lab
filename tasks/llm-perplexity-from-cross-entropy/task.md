## Context

Cross‑entropy loss measures the difference between a target distribution and a model’s predicted distribution. For a single sample with true class $t$ and logits $\mathbf{z}\in\mathbb R^V$, the probability assigned to $t$ is
$$p_t = \frac{\exp(z_t)}{\sum_{v=1}^{V} \exp(z_v)}.$$
The cross‑entropy for that sample is then
$$\ell = -\log p_t.$$

The perplexity of a language model over a dataset of $N$ samples is the exponential of the average cross‑entropy:
$$\mathrm{PP} = \exp\!\left(\frac1N\sum_{i=1}^{N}\ell_i\right).$$
A lower perplexity indicates that the model assigns higher probability to the true tokens.

## Task

Implement `perplexity_from_cross_entropy(logits, targets)`:

```python
def perplexity_from_cross_entropy(logits: list[list[float]],
                                   targets: list[int]) -> float:
    ...
```

`logits` is a 2‑D list of shape `(N, V)` containing the raw scores for each token in the vocabulary. `targets` is a 1‑D integer array of length `N` with the index of the correct token for each sample.

The function must compute the cross‑entropy loss per sample using the softmax of the logits, average over all samples, exponentiate that mean, and return the resulting perplexity as a Python float (or Python scalar). No explicit loops are allowed; use vectorised Python operations only.

## Example

```python
logits = [[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]]
targets = [0, 1]
pp = perplexity_from_cross_entropy(logits, targets)
print(pp)   # ≈ 2.13
```

## What the gate checks

The grader evaluates your implementation against a Python reference and reports the relative error
$$\mathrm{rel\_err}=\frac{|\,\hat{\mathrm{PP}}-\mathrm{PP}\,|}{|\mathrm{PP}|+10^{-12}}.$$
Your solution must achieve $\mathrm{rel\_err}\le 1\times10^{-6}$.

Additionally, the returned value must be a scalar of type `float`. Any other return type causes the gate to fail.
