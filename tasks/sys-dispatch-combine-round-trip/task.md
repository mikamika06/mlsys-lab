## Context

A Mixture-of-Experts (MoE) layer replaces one dense feed-forward block with
$E$ smaller "expert" transforms and a router that sends each token to one
(or a few) of them. On real hardware this is implemented as a three-stage
**dispatch → per-expert compute → combine** pipeline, because a single
expert's weight matrix is only ever multiplied against the tokens routed to
it:

1. **Dispatch** — group/permute the $n$ tokens by their assigned expert, so
   that every expert sees one contiguous batch of its own tokens.
2. **Expert compute** — apply expert $e$'s transform to its batch:
   $z_i = x_i W_{\,e(i)}$ for every token $i$ routed to expert $e(i)$.
3. **Combine** — scatter the per-expert outputs back into the original
   token order and scale each one by its router gate weight $g_i$:
   $y_i = g_i \cdot z_i = g_i \cdot \big(x_i W_{\,e(i)}\big)$.

For top-1 routing (each token goes to exactly one expert, no dropped
tokens), the whole pipeline is mathematically equivalent to the simple dense
form
$$
y_i = g_i \, \big(x_i \, W_{e(i)}\big), \qquad i = 1,\dots,n ,
$$
where $x_i \in \mathbb{R}^d$ is token $i$'s embedding, $e(i) \in
\{0,\dots,E-1\}$ is its assigned expert, $g_i \in \mathbb{R}$ its gate
weight, and $W_e \in \mathbb{R}^{d\times d}$ is expert $e$'s weight matrix.
The permute-compute-permute-back dance only matters for *how* the compute is
batched on hardware — the numerical result must match this dense form
exactly.

## Task

Implement:

```python
def moe_dispatch_combine(X: list[list[float]], expert_idx: list[int], gate_weight: list[float], W: list[list[list[float]]]) -> list[list[float]]:
    ...
```

* `X` — float array of shape $(n, d)$, the token embeddings.
* `expert_idx` — integer array of shape $(n,)$, `expert_idx[i]` is the
  expert index ($0 \le \text{expert\_idx}[i] < E$) that token $i$ is routed
  to.
* `gate_weight` — float array of shape $(n,)$, the combine weight for each
  token.
* `W` — float array of shape $(E, d, d)$, the per-expert weight matrices.

Implement the dispatch/combine round trip: group the tokens by
`expert_idx` (dispatch), multiply each expert's group of tokens by that
expert's matrix (expert compute), then scatter the results back into the
original token order and scale by `gate_weight` (combine). Return the
resulting array of shape $(n, d)$.

## Example

```python
X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
expert_idx = [0, 1, 0]
gate_weight = [1.0, 1.0, 0.5]
W = [
    [[2.0, 0.0], [0.0, 2.0]],   # expert 0: scale by 2
    [[1.0, 0.0], [0.0, -1.0]],  # expert 1: flip sign of 2nd coord
]

moe_dispatch_combine(X, expert_idx, gate_weight, W)
# token 0 -> expert 0: [1,0] @ W[0] = [2, 0],  * gate 1.0 -> [2, 0]
# token 1 -> expert 1: [0,1] @ W[1] = [0,-1],  * gate 1.0 -> [0,-1]
# token 2 -> expert 0: [1,1] @ W[0] = [2, 2],  * gate 0.5 -> [1, 1]
# -> array([[2., 0.], [0., -1.], [1., 1.]])
```

## What the gate checks

A single gate, **max_abs_err**, generates several random `(X, expert_idx,
gate_weight, W)` instances (varying $n$, $d$, $E$), computes the dense
reference $y_i = g_i (x_i W_{e(i)})$ directly for every token with Python,
and compares it element-wise to your function's output. The maximum
absolute error across all trials must be $\le 10^{-5}$; any exception or
wrong output shape counts as a failing (`1e9`) error.
