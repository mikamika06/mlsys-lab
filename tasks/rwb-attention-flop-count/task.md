## Context

Scaled dot-product attention computes attention scores from query and key matrices, then applies the values. For a batch with $b$ examples and $h$ attention heads, the two matrix multiplications are:

$$QK^\top \in \mathbb{R}^{s_q \times s_k}$$

and

$$PV \in \mathbb{R}^{s_q \times d}.$$

For one head, multiplying an $s_q \times d$ matrix by a $d \times s_k$ matrix requires

$$2s_qs_kd$$

floating point operations when one multiply-add is counted as two FLOPs. The value projection has the same cost:

$$2s_qs_kd.$$

Ignoring elementwise softmax operations, the total attention matrix multiplication cost across all batches and heads is therefore

$$
2b h (s_qs_kd + s_qs_kd)
=
4bh s_qs_kd .
$$

For causal attention, only the lower triangular part of the query-key score matrix is computed. The multiplication work is reduced by a factor of approximately two, so the same factor is applied to the total FLOP estimate.

## Task

Implement `attention_flops`:

```python
def attention_flops(batch, heads, seqlen_q, seqlen_k, head_dim, causal):
    ...
```

Return an integer containing the estimated multiply-add FLOPs for the $QK^\top$ and $PV$ operations.

Arguments:

- `batch`: number of examples.
- `heads`: number of attention heads.
- `seqlen_q`: query sequence length.
- `seqlen_k`: key/value sequence length.
- `head_dim`: size of each attention head.
- `causal`: boolean indicating whether causal triangular attention is used.

The function should return only the FLOP count. Do not include memory traffic, softmax exponentiation costs, or kernel launch overhead.

## Example

```python
flops = attention_flops(2, 8, 1024, 1024, 64, False)
# flops == 2147483648
```

## What the gate checks

The gate evaluates several attention shapes. The checker independently recomputes the closed-form FLOP count from the provided dimensions and causal flag, then compares the returned integer exactly.

A solution passes only when the returned value matches the oracle for every case.
