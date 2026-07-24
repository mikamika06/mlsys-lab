## Context

$L_0$ regularization ("how many weights are nonzero") is not
differentiable, so structured-pruning methods relax each binary gate into
a **hard-concrete** random variable: a stretched, hard-clipped sigmoid
whose *probability of being exactly zero* has a clean closed form. This
lets a network be trained with gradient descent to directly minimize an
expected sparsity, then pruned deterministically at test time.

Each gate is parameterized by a learnable $\log\alpha$ (the "location"),
a fixed temperature $\beta$, and fixed stretch bounds $\gamma < 0 < \zeta$
(paper defaults $\gamma=-0.1,\ \zeta=1.1$).

### Deterministic test-time gate value

At test time (no sampling), the gate collapses to:

$$
\hat z = \mathrm{clip}\Big(\mathrm{sigmoid}(\log\alpha)\cdot(\zeta-\gamma) + \gamma,\ 0,\ 1\Big)
$$

### Expected L0 (probability the gate survives)

The hard-concrete construction gives a closed form for
$P(\hat z_{\text{sampled}} > 0)$ — the probability mass that the gate
would be nonzero under the full stochastic (training-time) distribution:

$$
P(z>0) = \mathrm{sigmoid}\!\left(\log\alpha - \beta\log\!\left(\frac{-\gamma}{\zeta}\right)\right)
$$

Summed (or averaged) over all gates, this is the differentiable proxy for
the network's $L_0$ norm that gets added to the training loss.

## Task

Implement `hard_concrete_gate`:

```python
def hard_concrete_gate(log_alpha: np.ndarray, beta: float, gamma: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `log_alpha` — 1-D array of per-gate location parameters.
* `beta` — temperature (scalar float).
* `gamma`, `zeta` — stretch bounds, `gamma < 0 < zeta` (scalars).

Return `(gate_value, expected_l0)`, each the same shape as `log_alpha`,
computed exactly as defined above.

## Example

```python
import numpy as np
log_alpha = np.array([-2.0, 0.0, 2.0])
gate, l0 = hard_concrete_gate(log_alpha, beta=2/3, gamma=-0.1, zeta=1.1)
# gate in [0, 1]; l0 in (0, 1), increasing in log_alpha
```

## What the gate checks

* **gate_rel_err** — relative error between your `gate_value` and a
  NumPy oracle computing the deterministic test-time formula above, over
  several random `log_alpha` vectors and `(beta, gamma, zeta)` settings.
* **l0_rel_err** — relative error between your `expected_l0` and the
  oracle's closed-form $P(z>0)$, on the same trials.
