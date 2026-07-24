## Context

Multi‑head attention (MHA) splits the model dimension $d_{\text{model}}$ into $H$ heads, each of size  
$$d_k = \frac{d_{\text{model}}}{H}.$$  
For every head we compute three linear projections:
\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
\]
with $W_Q,W_K,W_V \in \mathbb{R}^{\,d_{\text{model}}\times d_k}$.
The scaled dot‑product attention for a head is
\[
A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,
\]
and the outputs of all heads are concatenated and projected back to $d_{\text{model}}$:
\[
Y = A_{\text{cat}} W_O,\qquad W_O \in \mathbb{R}^{\,H d_k\times d_{\text{model}}}.
\]

Ignoring bias terms, the parameter count for MHA is
\[
P_{\text{MHA}}
= 3\,d_{\text{model}}\,d_k + d_{\text{model}}^2.
\]
A single‑head attention with the same $d_{\text{model}}$ has
\[
P_{\text{single}}
= 4\,d_{\text{model}}^2,
\]
because each of the three projections is now square and the output projection is also square.

For a sequence of length $n$, the dominant floating‑point operations (FLOPs) are:

* per head: two matrix multiplications of size $(n\times d_k)\times(d_k\times n)$,  
  giving $4\,n^2 d_k$ FLOPs;
* one final projection of size $(n\times H d_k)\times(H d_k\times d_{\text{model}})$,  
  giving $2\,n\,d_{\text{model}}^2$ FLOPs.

Thus
\[
F_{\text{MHA}}
= H \cdot 4\,n^2 d_k + 2\,n\,d_{\text{model}}^2,
\]
and for the single‑head case
\[
F_{\text{single}}
= 4\,n^2 d_{\text{model}} + 2\,n\,d_{\text{model}}^2.
\]

## Task

Implement the function `compare_mha_vs_single` that takes three integers:

```python
def compare_mha_vs_single(d_model: int, heads: int, seq_len: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    ...
```

It must return a tuple of two tuples:
* `(params_mha, params_single)` – the integer parameter counts for MHA and single‑head attention.
* `(flops_mha, flops_single)` – the integer FLOP counts for the same.

All arithmetic should use Python integers; no floating point is required.  
The function must work for any positive integers where `d_model` is divisible by `heads`.

## Example

```python
>>> compare_mha_vs_single(64, 8, 32)
((8192, 16384), (262144, 524288))
```

Explanation:  
$d_k = 64/8 = 8$;  
$P_{\text{MHA}} = 3\times64\times8 + 64^2 = 8192$,  
$P_{\text{single}} = 4\times64^2 = 16384$.  
With $n=32$:  
$F_{\text{MHA}} = 8 \times 4 \times 32^2 \times 8 + 2 \times 32 \times 64^2 = 262144$,  
$F_{\text{single}} = 4 \times 32^2 \times 64 + 2 \times 32 \times 64^2 = 524288$.

## What the gate checks

The grader verifies that the returned tuple exactly matches a reference
calculation for several random test cases.  
The metric used is `exact_match`; any deviation causes failure.
