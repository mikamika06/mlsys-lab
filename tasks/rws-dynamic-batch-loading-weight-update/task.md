## Context

In online learning over a finite set of domains, we maintain a probability vector $w \in \Delta^{n-1}$ that represents how much training data to sample from each domain. After observing an excess loss vector $\ell \in \mathbb{R}^n$, the exponentiated‑gradient rule updates the weights by

$$
\tilde w_i = w_i\,\exp(\eta\,\ell_i), \qquad
w'_i = \frac{\tilde w_i}{\sum_{j=1}^{n}\tilde w_j},
$$

where $\eta>0$ is a learning rate. This rule preserves the simplex constraint and gives larger weight to domains with smaller loss.

## Task

Implement `update_weights(prior, excess, eta)` that performs this update. The function must accept any 1‑D array‑like objects for `prior` and `excess`, use NumPy vectorized operations only, return a NumPy array of dtype `float64`, and guarantee that the output sums to one.

## Example

```python
import numpy as np
prior = np.array([0.5, 0.3, 0.2])
excess = np.array([-0.1, 0.05, 0.02])
eta = 0.1
new_weights = update_weights(prior, excess, eta)
# new_weights ≈ [0.497, 0.298, 0.205]
```

## What the gate checks

The grader compares your result to a NumPy reference using the relative L2 error

$$
\mathrm{rel\_err} = \frac{\lVert w'_{\text{cand}}-w'_{\text{ref}}\rVert}
{\lVert w'_{\text{ref}}\rVert + 10^{-12}},
$$

requiring $\mathrm{rel\_err}\le 10^{-6}$, and verifies that the returned vector sums to one within $10^{-12}$. A correct implementation will pass both gates.
