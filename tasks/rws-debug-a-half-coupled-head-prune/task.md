## Context

Multi-head attention projects an input matrix $X \in \mathbb{R}^{n \times d}$ into query, key,
and value representations. For $h$ heads with head dimension $d_h = d / h$, the
projected tensors are split into head blocks.

For a selected set of heads $S$, pruning removes the corresponding slices from
the query, key, and value projections. The output projection must be coupled with
this change because its input dimension is the concatenation of head outputs.

If the kept heads are concatenated into $Z \in \mathbb{R}^{n \times (|S|d_h)}$,
the final projection is

$$Y = Z W_o^{S},$$

where $W_o^{S}$ contains only the input columns belonging to the kept heads.
Keeping the original $W_o$ creates a mismatch because the intermediate feature
dimension has changed.

## Task

Implement `pruned_attention_forward(x, q_proj, k_proj, v_proj, o_proj, heads, keep_heads)`.

Inputs:

- `x`: NumPy array with shape $(n, d)$.
- `q_proj`, `k_proj`, `v_proj`, `o_proj`: NumPy arrays of attention projection weights.
- `heads`: the original number of attention heads.
- `keep_heads`: a list of head indices that remain after pruning.

The function must:

1. Project $X$ using the query, key, and value weights.
2. Select only the requested head slices from the projected tensors.
3. Compute scaled dot-product attention for the remaining heads:

$$A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right),$$

$$Z = AV.$$

4. Concatenate the kept head outputs.
5. Apply the output projection using only the matching input columns of
   `o_proj`.

Return the final NumPy array.

## Example

```python
import numpy as np

x = np.zeros((2, 4))
q = np.eye(4)
k = np.eye(4)
v = np.eye(4)
o = np.eye(4)

y = pruned_attention_forward(x, q, k, v, o, 2, [0])
# y has shape (2, 2) because one of two heads remains
```

## What the gate checks

The gate builds random attention projections and computes an oracle result using
a NumPy implementation of correctly coupled head pruning. The returned value is
compared with the oracle using maximum absolute error.

The gate passes when

$$\max_i |y_i - y_i^{oracle}| < 10^{-6}.$$

A solution that prunes query, key, and value heads but applies the original
unpruned output projection fails because the output projection no longer matches
the reduced attention representation.
