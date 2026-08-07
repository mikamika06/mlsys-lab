## Context

Tensor parallelism splits the computation of a neural network layer across multiple
devices. In Megatron-style attention parallelism, attention heads are partitioned
between ranks. Each rank computes attention for only its assigned heads, then the
partial outputs are combined.

For queries $Q$, keys $K$, and values $V$, scaled dot-product attention for one
head is

$$
\mathrm{Attn}(Q,K,V) =
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

A multi-head attention layer computes this independently for $h$ heads. If the
attention output is concatenated as $O \in \mathbb{R}^{b \times s \times hd}$ and
the output projection is $W_o \in \mathbb{R}^{hd \times m}$, the final result is

$$
Y = O W_o .
$$

When heads are sharded across ranks, rank $r$ computes a subset of rows of
$W_o$ and produces a partial contribution

$$
Y_r = O_r W_{o,r}.
$$

The full tensor-parallel result is recovered by an all-reduce sum:

$$
Y = \sum_r Y_r .
$$

## Task

Implement `sharded_attention_heads`:

```python
def sharded_attention_heads(q: list[list[list[list[float]]]], k: list[list[list[list[float]]]], v: list[list[list[list[float]]]], wo: list[list[float]], num_ranks: int) -> list[list[list[float]]]:
    ...
```

The inputs are list:

- `q`, `k`, and `v` have shape $(b, h, s, d)$, where $b$ is batch size,
  $h$ is the number of attention heads, $s$ is sequence length, and $d$ is the
  head dimension.
- `wo` has shape $(h d, m)$ and is the output projection matrix.
- `num_ranks` is the number of tensor-parallel ranks.

Return a list with shape $(b, s, m)$ containing the same result as a
single-device multi-head attention implementation.

Split the heads evenly between ranks. For each rank, compute attention for its
heads, multiply by the matching rows of `wo`, and sum the rank contributions.
Use Python operations only.

## Example

```python

q = [[[[0.0], [0.0]], [[0.0], [0.0]]]]
k = [[[[0.0], [0.0]], [[0.0], [0.0]]]]
v = [[[[1.0], [2.0]], [[3.0], [4.0]]]]
wo = [[1.0] * 1 for _ in range(2)]

y = sharded_attention_heads(q, k, v, wo, 2)
# shape is (1, 2, 1)
```

## What the gate checks

The gate computes a single-device Python reference implementation of
multi-head attention and compares it with the submitted tensor-parallel
implementation.

The maximum absolute difference

$$
\max_i |y_i - \hat{y}_i|
$$

must be less than $10^{-5}$. A solution that does not correctly partition heads,
apply the output projection shards, and perform the equivalent all-reduce will
fail.
