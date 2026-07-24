## Context

Attention implementations often represent causal masking as an additive logit bias. For
a query position $i$ and key position $j$, a causal attention mask prevents a query
from attending to future keys:

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

Adding this matrix to attention logits is equivalent to enabling a causal flag in a
kernel that applies the same restriction internally.

Some attention variants also add ALiBi (Attention with Linear Biases). Given a slope
$s$, the additive bias is

$$
B_{ij} = s(j-i),
$$

where the key index is $j$ and the query index is $i$. This shifts logits while
preserving the causal masking behavior.

## Task

Implement `apply_attention_bias`:

```python
def apply_attention_bias(logits, is_causal=False, alibi_slope=None):
    ...
```

The input `logits` is a 2-D NumPy array with shape $(q, k)$ containing attention
scores. Return a `float64` NumPy array.

When `is_causal=True`, add the explicit causal additive mask where entries above the
diagonal become negative infinity.

When `alibi_slope` is not `None`, add the ALiBi bias
$s(j-i)$ to every position. The ALiBi bias is added independently from the causal
mask. If both options are enabled, both additive terms must be applied.

Do not modify the input array in place.

## Example

```python
import numpy as np

logits = np.zeros((3, 3), dtype=np.float32)
out = apply_attention_bias(logits, is_causal=True, alibi_slope=0.5)

# The upper triangle contains -inf from the causal mask.
# The finite entries include the ALiBi shift.
```

## What the gate checks

The gate builds an independent NumPy oracle. It compares the causal output against a
reference that constructs the explicit upper-triangular $-\infty$ mask. It also
compares ALiBi output against the analytic bias formula
$B_{ij}=s(j-i)$.

The maximum absolute error of the causal result must be below $10^{-6}$ and the
maximum absolute error of the ALiBi result must be below $10^{-5}$.
