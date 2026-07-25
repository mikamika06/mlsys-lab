## Context

Attention Linear Bias (ALiBi) augments the attention logits with a linear bias that depends on the distance between query and key positions.  
For head $h$ the bias is

$$b_h(d)=\text{slope}_h \cdot d,$$

where $d$ is the relative position (negative for earlier tokens).  The slopes are chosen to decay exponentially across heads so that early heads focus on short‑range dependencies while later heads attend more globally.  
The closed‑form formula used in the original ALiBi paper is

$$\text{slope}_h = 2^{-(k+3-h)} \quad (0\le h<k),$$

where $k$ is the number of heads that are a power of two.  For arbitrary $n$, the slopes for the first $m=2^{\lfloor\log_2 n\rfloor}$ heads follow this formula, and the remaining $n-m$ slopes are obtained recursively.

## Task

Implement `alibi_slopes(n_heads)`:

```python
def alibi_slopes(n_heads: int) -> np.ndarray:
    ...
```

The function must:

1. Accept a positive integer `n_heads`.
2. Return a NumPy array of shape `(n_heads,)` containing the slopes for each head.
3. Use the closed‑form algorithm described above, handling any `n_heads`, not just powers of two.
4. Produce an array with dtype `np.float32`.

## Example

```python
import numpy as np
from your_module import alibi_slopes

slopes = alibi_slopes(4)
print(slopes)  # [0.0078125, 0.015625 , 0.03125  , 0.0625   ]
```

For `n_heads=4` the slopes are `[2^{-7}, 2^{-6}, 2^{-5}, 2^{-4}]`.

## What the gate checks

The grader computes a reference array using the exact algorithm and compares it to your output with the relative error metric:

$$\texttt{rel\_err} = \frac{\lVert \hat{s}-s\rVert_2}{\lVert s\rVert_2 + 10^{-12}}.$$

Your solution must achieve `rel_err ≤ 1e-6`.  Any deviation larger than this threshold will cause the gate to fail.  The grader also verifies that your function accepts only positive integers and returns a NumPy array of dtype `float32`.
