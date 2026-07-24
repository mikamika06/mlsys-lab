## Context

An autoregressive language model produces, at every decode step $t$, a vector of
logits $\mathbf{z}^{(t)} \in \mathbb{R}^{V}$ over a vocabulary of size $V$.
Temperature sampling turns each logit vector into a probability distribution and
draws one token from it. With temperature $\tau > 0$,

$$
p^{(t)}_i = \frac{\exp(z^{(t)}_i / \tau)}{\sum_{j=1}^{V} \exp(z^{(t)}_j / \tau)} .
$$

To draw a token we use **inverse-CDF sampling**. Let the cumulative distribution
be $F^{(t)}_i = \sum_{j \le i} p^{(t)}_j$. Given a uniform draw $u \in [0, 1)$ the
sampled token is the smallest index $i$ whose cumulative mass exceeds $u$:

$$
\text{id}^{(t)} = \min\{\, i : F^{(t)}_i > u \,\} .
$$

Reproducibility comes from the random stream. All draws come from a **single**
`numpy` generator created once as `np.random.default_rng(seed)`, and one uniform
$u$ is consumed **per step, in order** via `rng.random()`. Because the generator
state advances deterministically, the whole sampled id sequence is a pure
function of `logits`, `temperature`, and `seed`. As $\tau \to 0^{+}$ the
distribution collapses onto the argmax and sampling reproduces greedy decoding.

## Task

Implement a function with this exact signature:

```python
def sample_sequence(logits: np.ndarray, temperature: float, seed: int) -> np.ndarray:
    ...
```

* `logits` — a 2-D `float64` array of shape $(T, V)$: one logit row per decode step.
* `temperature` — a positive float $\tau$.
* `seed` — an integer seed for `np.random.default_rng`.

Follow this exact convention so the sampled sequence is reproducible:

1. Create the generator once: `rng = np.random.default_rng(seed)`.
2. For each step $t = 0, 1, \dots, T-1$ **in order**:
   1. scale the row: $\mathbf{z} = \text{logits}[t] / \tau$;
   2. softmax it into probabilities $\mathbf{p}$ (subtract the row max for numerical
      stability before exponentiating);
   3. form the cumulative distribution `cdf = np.cumsum(p)`;
   4. draw a single uniform `u = rng.random()`;
   5. pick the smallest index $i$ with $\text{cdf}[i] > u$
      (equivalently `np.searchsorted(cdf, u, side="right")`);
   6. clamp the index to `V - 1` in case floating-point rounding leaves
      $u \ge \text{cdf}[-1]$.

Return the sampled ids as a 1-D `int64` array of shape $(T,)$.

## Example

```python
import numpy as np

logits = np.array([[2.0, 0.0, -1.0],
                   [0.5, 0.5,  0.5]])
ids = sample_sequence(logits, temperature=1.0, seed=0)
print(ids)          # e.g. array([0, 2])  (exact values depend on the stream)
```

Row 0 is peaked on token 0, so it is sampled most of the time; row 1 is uniform,
so any of the three tokens can appear. The exact ids are fixed once `seed` is fixed.

## What the gate checks

The grader builds several random `(logits, temperature, seed)` cases and recomputes
the reference id sequence with NumPy using the exact convention above — nothing is
hard-coded. Your output must match the reference id-for-id (same shape, `int64`
dtype) across every case. The metric **exact_match** must equal `1.0`; any
mismatch, wrong dtype/shape, drawing the wrong number of uniforms, or consuming the
generator out of order yields `0.0`.
