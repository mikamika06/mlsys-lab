## Context

Temperature scaling is a post‑processing technique used to calibrate the confidence of a classifier. Given logits $z \in \mathbb{R}^n$ produced by a neural network, we first divide them by a temperature parameter $T>0$, then apply the softmax function:
$$
p_i = \frac{\exp(z_i/T)}{\sum_{j=1}^{n}\exp(z_j/T)} .
$$

When $T=1$ this is the ordinary softmax. Larger temperatures produce flatter distributions, while smaller temperatures sharpen them.

## Task

Implement `temperature_scale(logits: list[float], T: float) -> list[float] that returns the temperature‑scaled probability distribution for a 1‑D array of logits. The function must:

- Accept any list of shape `(n,)` or `(batch,n)` and return an array of the same shape.
- Use only Python operations; no explicit Python loops.
- Return a `float64` array.

## Example

```python
logits = [2.0, 1.0, 0.1]
T = 0.5
probs = temperature_scale(logits, T)
print(probs)  # [0.8637771182080067, 0.11689952094598567, 0.019323360846007505]
```

## What the gate checks

The grader computes a reference distribution using Python’s softmax with the same temperature and measures the mean Kullback–Leibler divergence
$$
\text{mean\_kl} = \frac1n \sum_{i=1}^{n}
p_i^{\text{ref}}\,
\log\!\left(\frac{p_i^{\text{ref}}}{p_i^{\text{cand}}}\right).
$$

The candidate must achieve $\text{mean\_kl} \le 10^{-6}$ against the reference. A correct implementation will produce a distribution that matches Python’s result up to machine precision.
