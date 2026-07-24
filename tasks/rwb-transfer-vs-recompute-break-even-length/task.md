## Context

When a large language model serves a long prompt, the system must decide
whether to **transfer** the already-computed key-value (KV) cache from the
prefill node to the decode node, or to **recompute** the prefill on the decode
node from scratch.

Let $b$ be the number of bytes in the KV cache per token (across all layers
and heads), $\beta$ the network bandwidth in bytes/s, $\lambda$ the fixed
network latency (round-trip) in seconds, and $\rho$ the prefill throughput of
the decode node in tokens/s. For a prompt of $L$ tokens the two costs are:

$$
T_{\text{transfer}}(L) = \lambda + \frac{L \cdot b}{\beta},
\qquad
T_{\text{recompute}}(L) = \frac{L}{\rho}.
$$

Transfer has a constant latency term $\lambda$ but a per-token slope
$b / \beta$.  Recomputation has zero fixed cost but a steeper per-token slope
$1 / \rho$ (assuming the network is faster per byte than the GPU's arithmetic
throughput).  The **break-even length** is the smallest $L^*$ such that
$T_{\text{transfer}}(L^*) = T_{\text{recompute}}(L^*)$:

$$
\lambda + \frac{L^* \, b}{\beta} = \frac{L^*}{\rho}
\quad\Longrightarrow\quad
L^* = \frac{\lambda}{\dfrac{1}{\rho} - \dfrac{b}{\beta}}
= \frac{\lambda \, \rho \, \beta}{\beta - b \, \rho}.
$$

This formula is valid only when $\beta > b\,\rho$, i.e.\ the network can ship
each token's KV bytes faster than the GPU can produce them.  If
$\beta \le b\,\rho$, recomputation is always at least as fast regardless of
sequence length.

## Task

Implement `break_even_length`:

```python
def break_even_length(
    kv_bytes_per_token: float,
    bandwidth: float,
    latency: float,
    prefill_throughput: float,
) -> float:
```

Parameters:

| Name | Description |
|---|---|
| `kv_bytes_per_token` $b$ | Total KV-cache bytes per token across all layers |
| `bandwidth` $\beta$ | Network bandwidth in bytes/s |
| `latency` $\lambda$ | Fixed network round-trip latency in seconds |
| `prefill_throughput` $\rho$ | Decode-node prefill throughput in tokens/s |

Return the break-even sequence length $L^*$ as a `float`.  If recomputation
is never slower than transfer ($\beta \le b\,\rho$), return `float('inf')`.
If the latency is zero and transfer is strictly cheaper per token, return
`0.0`.

All inputs are positive except `latency` which is non-negative.

## Example

```python
# b = 4096 B/tok, beta = 1 GB/s, lambda = 1 ms, rho = 10 000 tok/s
L = break_even_length(4096, 1e9, 0.001, 10_000)
# L ≈ 11.43 tokens
# Transfer: 0.001 + 11.43 * 4096 / 1e9 = 0.001 + 0.0000468 ≈ 0.0010468 s
# Recompute: 11.43 / 10000 ≈ 0.001143 s
```

```python
# When bandwidth is huge, transfer wins for any L > 0
L = break_even_length(100, 1e12, 0.001, 10_000)
# L ≈ 0.1001 tokens (essentially zero — transfer always wins)
```

```python
# When latency is zero and transfer is faster per token, return 0.0
L = break_even_length(100, 1e10, 0.0, 15_000)
# L = 0.0
```

## What the gate checks

A single gate.  The grader recomputes the break-even length from the closed-form
solution using the same formula on eight diverse test vectors (varying
bandwidth from $5 \times 10^8$ to $10^{12}$ B/s, latency from $0$ to $3$ ms,
throughput from $100$ to $15\,000$ tok/s, and KV bytes from $100$ to $8\,192$
B/tok).  It returns the maximum relative error across all cases:

$$
\text{rel\_err} = \max_i \frac{|L^*_i - \hat{L}^*_i|}{|L^*_i| + \epsilon}
$$

with $\epsilon = 10^{-300}$ for numerical safety (and special handling when
$L^* = \infty$).  The gate opens when $\text{rel\_err} < 10^{-6}$.
