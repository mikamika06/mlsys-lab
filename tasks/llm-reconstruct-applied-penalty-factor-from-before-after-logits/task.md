## Context

Language models can adjust token logits before sampling. A multiplicative penalty
factor can be applied to selected token logits. For an original logit $x$ and a
penalized logit $y$, the relationship is:

$$
y = \frac{x}{p}.
$$

Rearranging gives an estimate of the applied penalty:

$$
p = \frac{x}{y}.
$$

Real systems may provide several affected tokens. Their individual estimates can
differ slightly because of numerical effects or token-specific processing, so the
combined factor should be reconstructed in log space:

$$
p = \exp\left(\frac{1}{m}\sum_{i=1}^{m}\log\left(\frac{x_i}{y_i}\right)\right),
$$

where $m$ is the number of affected tokens.

## Task

Implement `reconstruct_penalty_factor`:

```python
def reconstruct_penalty_factor(before, after, affected_indices):
    ...
```

Arguments:

- `before`: a 1-D NumPy array containing logits before the penalty.
- `after`: a 1-D NumPy array containing logits after the penalty.
- `affected_indices`: integer indices of tokens whose logits were modified.

Return the reconstructed penalty factor as a Python `float`.

Only the selected indices should be used. The affected logits are positive finite
values.

## Example

```python
import numpy as np

before = np.array([4.0, 2.0, 8.0])
after = np.array([2.0, 1.0, 8.0])

p = reconstruct_penalty_factor(
    before,
    after,
    np.array([0, 1])
)
# p is 2.0
```

## What the gate checks

The gate generates logits, applies penalty factors to affected tokens, and computes
the reference reconstruction with the log-space formula. The candidate result is
compared using:

$$
\mathrm{rel\_err} =
\frac{|p_{\mathrm{candidate}} - p_{\mathrm{reference}}|}
{|p_{\mathrm{reference}}| + 10^{-12}}.
$$

The value must satisfy $\mathrm{rel\_err} < 10^{-4}$.
