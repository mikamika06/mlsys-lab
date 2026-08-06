## Context

Multi-head attention (MHA) stores separate key and value projections for every
attention head. Grouped-query attention (GQA) reduces the number of key/value
heads while keeping multiple query heads.

Suppose an MHA key tensor has shape $(B, H, T, d)$, where $H$ is the number of
query/key heads. GQA uses $G$ key/value heads with $G < H$. A common uptraining
initialization creates each new key/value head by averaging the original heads
inside a contiguous group.

For a group containing key heads $K_1, \dots, K_m$, the pooled head is

$$
K_{\mathrm{new}} = \frac{1}{m}\sum_{i=1}^{m}K_i .
$$

The same operation is applied to value heads. This reduces the number of KV
heads while preserving the average information from the original MHA weights.

The conversion can be checked by treating the pooled tensors as distributions
over their final dimension. The KL divergence is

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}.
$$

A correct implementation produces exactly the oracle pooled tensors, giving a
zero KL divergence against the reference representation.

## Task

Implement `uptrain_mha_to_gqa(q, k, v, groups)`:

```python
def uptrain_mha_to_gqa(q, k, v, groups):
    ...
```

The inputs have shapes:

- `q`: $(B, H, T_q, d)$ query tensor.
- `k`: $(B, H, T_k, d)$ MHA key tensor.
- `v`: $(B, H, T_k, d)$ MHA value tensor.
- `groups`: target number of GQA key/value heads.

Assume $H$ is divisible by `groups`. Return `(k_gqa, v_gqa)` with shapes
$(B, G, T_k, d)$.

Heads are assigned contiguously. If $H=8$ and $G=2$, heads $0$ through $3$
form one group and heads $4$ through $7$ form the second group.

Use Python operations only.

## Example

```python

q = [[[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]]
k = list(range(24)).reshape(1, 4, 2, 3)
v = k.copy()

kg, vg = uptrain_mha_to_gqa(q, k, v, 2)

# kg has shape (1, 2, 2, 3)
# kg[0, 0] is the mean of k[0, 0] and k[0, 1]
```

## What the gate checks

The gate generates MHA tensors and computes the reference GQA tensors with the
pooling algorithm itself. It compares the submitted tensors against that
reference using a KL divergence metric over the tensor values.

The metric is

$$
\mathrm{mean\_kl} =
D_{\mathrm{KL}}(\mathrm{softmax}(R)\Vert\mathrm{softmax}(C)),
$$

where $R$ is the oracle result and $C$ is the submitted result. The value must
satisfy $\mathrm{mean\_kl}\le 10^{-12}$.
