## Context

Speculative decoding uses a cheaper draft model to propose tokens and a target model to verify them. A key design question is whether a draft configuration provides enough accepted tokens to offset the additional draft computation.

For a draft model with per-step acceptance probability $\alpha$ and block length $k$, the expected number of accepted target tokens during one verification step is the finite geometric sum

$$
E = 1 + \alpha + \alpha^2 + \dots + \alpha^k
$$

which can also be written as

$$
E = \frac{1-\alpha^{k+1}}{1-\alpha}
$$

for $\alpha \ne 1$. When $\alpha = 1$, every proposed token is accepted and $E = k+1$.

Let $c$ be the draft-model cost relative to one target-model forward pass. A verification cycle costs

$$
1 + kc
$$

relative cost units. Compared with generating one token at a time with only the target model, the modeled speedup is

$$
S = \frac{E}{1+kc}.
$$

## Task

Implement `draft_speedup_model(configs)`:

```python
def draft_speedup_model(configs: np.ndarray) -> np.ndarray:
    ...
```

`configs` is a NumPy array of shape $(n, 3)`. Each row contains:

- `alpha`: the per-step acceptance rate, where $0 \le \alpha \le 1$
- `k`: the draft block length as an integer
- `c`: the draft-to-target cost ratio

Return a NumPy array of shape $(n, 2)`. Column $0$ must contain the expected accepted tokens per verification, and column $1$ must contain the modeled speedup.

Use floating point output.

## Example

```python
import numpy as np

configs = np.array([
    [0.5, 2, 0.2],
])

result = draft_speedup_model(configs)

# result[0, 0] = 1.75
# result[0, 1] = 1.25
```

The first value is

$$
1 + 0.5 + 0.5^2 = 1.75
$$

and the speedup is

$$
\frac{1.75}{1 + 2(0.2)} = 1.25.
$$

## What the gate checks

The gate builds several draft configurations and recomputes the expected values using the geometric-series model. It compares both returned columns against this oracle using relative error.

The implementation must handle normal acceptance rates and the boundary case $\alpha = 1$.
