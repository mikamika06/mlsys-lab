## Context

In a Mixture-of-Experts (MoE) layer, a router produces a probability vector
$g \in \mathbb{R}^{E}$ over $E$ experts via a softmax:

$$g_i = \frac{\exp(z_i)}{\sum_{j=1}^{E} \exp(z_j)}, \qquad \sum_{i=1}^{E} g_i = 1.$

Each expert $f_i$ transforms the input $x$ into a hidden-state vector
$f_i(x) \in \mathbb{R}^{d}$. The MoE output is the **gated** combination:

$$y = \sum_{i=1}^{E} g_i \cdot f_i(x) \in \mathbb{R}^{d}.$$

A common implementation bug is to discard or ignore the gate weights $g$
and instead compute an unweighted mean of the expert outputs:

$$\hat{y} = \frac{1}{E}\sum_{i=1}^{E} f_i(x).$$

When every expert returns the same output this produces the correct answer
by coincidence, which lets the bug hide during smoke tests with uniform
inputs. The error surfaces whenever experts diverge and gate probabilities
are non-uniform.

## Task

The file `starter.py` contains a function `moe_combine` with exactly one
bug.  **Find and fix it** so the function correctly applies the gate weights
to the expert outputs.

```python
def moe_combine(expert_outputs: np.ndarray, gate_weights: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    expert_outputs : (n_experts, d)  – each row is one expert's output vector
    gate_weights   : (n_experts,)    – softmax probabilities, sum to 1.0

    Returns
    -------
    (d,) – the weighted combination  sum_i gate_weights[i] * expert_outputs[i]
    """
```

Do **not** rename the function.  You may change only the body.

## Example

```python
import numpy as np
expert_outputs = np.array([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]])   # 3 experts, d=2
gate_weights   = np.array([0.0, 0.0, 1.0])  # only the third expert matters

y = moe_combine(expert_outputs, gate_weights)
# y == [5.0, 6.0]
```

With the buggy starter the function returns `[3.0, 4.0]` (the unweighted
mean), which is wrong.

## What the gate checks

A single gate: the relative $L^{2}$ error

$$\text{rel\_err} = \frac{\lVert \hat{y} - y^{*} \rVert}{\lVert y^{*} \rVert}$$

between your output $\hat{y}$ and the reference $y^{*}$ (computed from the
oracle inside `check.py`) must be below $10^{-5}$ across all test cases.
The test cases include non-uniform gate weights with divergent expert
outputs, so the unweighted-mean bug produces $\text{rel\_err} > 0.1$.
