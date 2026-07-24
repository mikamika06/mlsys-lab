## Context

Mixture-of-Experts (MoE) models route each token to one or more experts. A router
produces probabilities over $E$ experts. To prevent routing collapse, Switch
Transformer introduces an auxiliary load-balancing loss.

For a batch of $N$ routed tokens, let $f_i$ be the fraction of tokens assigned
to expert $i$:

$$
f_i = \frac{1}{N}\sum_{x=1}^{N}\mathbf{1}(\operatorname{argmax}(p_x)=i),
$$

where $p_x$ is the router probability vector for token $x$.

Let $P_i$ be the mean router probability assigned to expert $i$:

$$
P_i = \frac{1}{N}\sum_{x=1}^{N}p_{x,i}.
$$

The auxiliary loss is

$$
L_{\mathrm{aux}} = E \sum_{i=1}^{E} f_i P_i .
$$

A balanced router has similar values for all experts, while a collapsed router
has a larger loss because both assignment frequency and probability mass are
concentrated.

## Task

Implement `load_balancing_aux_loss(router_probs)`:

```python
def load_balancing_aux_loss(router_probs: np.ndarray) -> float:
    ...
```

The input is a 2-D NumPy array of shape $(N, E)$ containing non-negative router
probabilities. Each row represents one token and sums to $1$. Return the scalar
Switch-Transformer load-balancing auxiliary loss as a Python `float`.

Use NumPy operations for the computation. The returned value should be computed
from both the discrete routing decision (the per-row maximum expert) and the
router probability averages.

## Example

```python
import numpy as np

router_probs = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.7, 0.2],
    [0.2, 0.2, 0.6],
])

loss = load_balancing_aux_loss(router_probs)
# loss is approximately 1.06
```

## What the gate checks

The gate builds several router probability matrices and computes the reference
answer using the mathematical definition of the Switch Transformer auxiliary
loss. The candidate result is compared with the oracle using relative error:

$$
\mathrm{rel\_err} =
\frac{|L_{\mathrm{candidate}}-L_{\mathrm{ref}}|}
{|L_{\mathrm{ref}}|+10^{-12}} .
$$

The relative error must be below $10^{-6}$. Implementations that use only the
mean probabilities, ignore expert assignment frequencies, or omit the expert
count factor will fail.
