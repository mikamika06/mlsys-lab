## Context

Multi-LoRA serving systems (S-LoRA, Punica) batch many concurrent
requests together even when each one uses a **different** LoRA adapter.
SGMV (Segmented Gather Matrix-Vector multiply) is the primitive that
makes this efficient: rows are grouped by which adapter they use, and
each group is processed with one batched matmul against that adapter's
factors, instead of falling back to one tiny matmul per row.

Given a bank of `num_adapters` LoRA adapters, all sharing rank $r$:

$$
A_{\text{bank}} \in \mathbb{R}^{\text{num\_adapters}\times d_{\text{in}}\times r}, \qquad
B_{\text{bank}} \in \mathbb{R}^{\text{num\_adapters}\times r\times d_{\text{out}}}, \qquad
\text{scale} \in \mathbb{R}^{\text{num\_adapters}}
$$

row $i$ of a batch $x \in \mathbb{R}^{N\times d_{\text{in}}}$, tagged with
adapter id $a_i$, produces

$$
y_i = \text{scale}[a_i] \cdot \big(x_i \, A_{\text{bank}}[a_i]\big)\, B_{\text{bank}}[a_i]
$$

Mathematically this is identical to just looping over rows and applying
each one's own adapter — SGMV is a *scheduling* trick (group rows by
adapter, batch the matmul per group), not a different numerical result.

## Task

Implement `sgmv_apply`:

```python
def sgmv_apply(x: np.ndarray, adapter_id: np.ndarray,
                A_bank: np.ndarray, B_bank: np.ndarray, scale: np.ndarray) -> np.ndarray:
    ...
```

- `x`: `(N, d_in)`.
- `adapter_id`: `(N,)` int, `adapter_id[i]` indexes into the bank for
  row `i`.
- `A_bank`: `(num_adapters, d_in, r)`.
- `B_bank`: `(num_adapters, r, d_out)`.
- `scale`: `(num_adapters,)` float.
- Return `(N, d_out)`, where row `i` is
  `scale[adapter_id[i]] * (x[i] @ A_bank[adapter_id[i]]) @ B_bank[adapter_id[i]]`.

## Example

```python
import numpy as np

x = np.random.default_rng(0).standard_normal((5, 4))
adapter_id = np.array([0, 1, 0, 1, 0])
A_bank = np.random.default_rng(1).standard_normal((2, 4, 2))
B_bank = np.random.default_rng(2).standard_normal((2, 2, 3))
scale = np.array([1.0, 0.5])

out = sgmv_apply(x, adapter_id, A_bank, B_bank, scale)
# out.shape == (5, 3)
# out[0], out[2], out[4] all used adapter 0; out[1], out[3] used adapter 1.
```

## What the gate checks

The grader loads a committed fixture (20 rows over 4 adapters, rank 3)
plus several additional seeded random batches, and compares your output
to an oracle that applies each row's own adapter independently, one row
at a time, in a plain Python loop — never calling your function, never
hardcoding an expected value, and structurally unable to leak one row's
adapter into another's output.

`max_abs_err` is the worst per-case max-abs-error across all cases and
must be `<= 1e-5`. Applying the wrong adapter to a row, sharing one
adapter's factors across the whole batch, or forgetting the per-adapter
`scale` will all produce a visible mismatch for at least one row.
