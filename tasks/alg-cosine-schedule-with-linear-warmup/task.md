## Context

In training neural networks we often vary the learning rate over time. A popular strategy is a cosine decay that smoothly reduces the step size from an initial value $\\alpha_0$ to a minimum $\\alpha_{\min}$, optionally preceded by a linear warm‑up phase that raises the rate from zero to $\\alpha_0$ during the first $w$ steps.

The cosine schedule for the non‑warm‑up part is

$$
\\alpha_t = \\alpha_{\min} + (\\alpha_0-\\alpha_{\min})\\,
\\frac{1+\\cos\\bigl(\\pi\\,t/T\\bigr)}{2},
$$

where $t$ runs from $0$ to $T-1$ and $T$ is the number of steps after warm‑up. The linear warm‑up is simply

$$
\\alpha_t = \\frac{t+1}{w}\\,\\alpha_0,
$$

for $t=0,\\dots,w-1$.

## Task

Implement a function that returns the learning rate for every step of a training run:

```python
def lr_schedule(total_steps: int,
                warmup_steps: int,
                base_lr: float,
                min_lr: float = 0.0) -> np.ndarray:
    ...
```

The returned array must have shape `(total_steps,)` and contain the learning rate for each step index $i$ (starting at $0$). The function should use only NumPy operations; no Python loops.

## Example

```python
import numpy as np
lrs = lr_schedule(total_steps=5, warmup_steps=2,
                  base_lr=0.1, min_lr=0.01)
print(lrs)
# [0.05  0.10  0.075 0.0375 0.0125]
```

Here the first two steps are a linear warm‑up from $0$ to $0.1$, and the remaining three follow a cosine decay down to $0.01$.

## What the gate checks

The grader computes a reference schedule with NumPy and compares it to your output using the metric `max_abs_err`. Your implementation must achieve an error not larger than $10^{-12}$, which is effectively bit‑exact for double precision arithmetic.
