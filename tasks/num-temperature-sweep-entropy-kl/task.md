## Context

The softmax function maps a vector of logits $z \in \mathbb{R}^n$ to a probability distribution over the $n$ classes:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}.$$

When a temperature parameter $T>0$ is introduced, the logits are scaled before exponentiation:

$$\operatorname{softmax}_T(z)_i = \frac{\exp(z_i/T)}{\sum_{j=1}^{n}\exp(z_j/T)}.$$

A larger temperature produces a softer distribution; as $T \to 0^+$ the softmax approaches an arg‑max.

The Kullback–Leibler divergence between two probability distributions $p$ and $q$ over the same support is

$$D_{\mathrm{KL}}(p\|q)=\sum_{i=1}^{n} p_i \log\frac{p_i}{q_i}.$$

Its mean over a set of temperature values measures how close a candidate implementation is to a reference.

## Task

Implement `softmax_temperature_sweep`:

```python
def softmax_temperature_sweep(logits: list[float], temps: Sequence[float]) -> list[list[float]]:
    ...
```

* `logits` – a 1‑D list of shape `(n,)` containing the raw logits for a single sample.
* `temps` – an iterable of positive floats representing temperatures to sweep over.

The function must return a 2‑D list of shape `(len(temps), n)` where each row contains the softmax probabilities computed with the corresponding temperature. The implementation must be numerically stable: subtract the maximum value of the scaled logits before exponentiation, and use `float64` arithmetic throughout.

## Example

```python
logits = [0., 1., 2.]
temps = [0.5, 1.0, 2.0]
probs = softmax_temperature_sweep(logits, temps)
print(probs)  # [[0.015876239976466765, 0.11731042782619838, 0.8668133321973349], [0.09003057317038046, 0.24472847105479764, 0.6652409557748218], [0.1863237232258476, 0.3071958857184984, 0.506480391055654]]
```

## What the gate checks

The grader computes a reference implementation using Python and compares it to your output with the mean Kullback–Leibler divergence `mean_kl`. The candidate passes only if

$$\text{mean\_kl} \le 10^{-9}.$$

A correct, numerically stable implementation will achieve this threshold. A naive or incorrect version (e.g., missing temperature scaling or wrong max‑subtraction) will produce a much larger divergence and fail the gate.
