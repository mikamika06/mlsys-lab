## Context

In a transformer layer the two main computational blocks are the multi‑head self‑attention sublayer and the feed‑forward network (FFN).  
For a sequence of length $L$ and model dimension $d_{\text{model}}$, the FLOPs required by each block can be approximated as follows.

**Self‑attention.**  
The dominant cost comes from computing the scaled dot‑product attention scores for every head.  
With $h$ heads, each head has key/query/value dimension $d_k=d_v=d_{\text{model}}/h$.  
For a single head we need $L^2 d_k$ multiplications to form the score matrix and an equal number of additions; the soft‑max and weighted sum add another $O(L^2 d_k)$ operations.  
Summed over all heads this yields

$$
\text{FLOPs}_{\text{attn}}(L) \;\approx\; 8\,L^{2}\,d_{\text{model}},
$$

where the constant $8$ absorbs both multiplications and additions.

**Feed‑forward network.**  
The FFN consists of two linear layers with hidden size $d_{\text{ff}}$.  
Per token we perform $d_{\text{model}}\times d_{\text{ff}}$ multiplications for the first layer and $d_{\text{ff}}\times d_{\text{model}}$ for the second, together with a comparable number of additions.  Thus

$$
\text{FLOPs}_{\text{FFN}}(L) \;\approx\; 4\,L\,d_{\text{model}}\,d_{\text{ff}},
$$

where $4$ counts both multiplications and additions.

The **crossover sequence length** is the smallest integer $L$ for which
$\text{FLOPs}_{\text{attn}}(L) > \text{FLOPs}_{\text{FFN}}(L)$.
Solving the inequality gives

$$
8\,L^{2}\,d_{\text{model}}
> 4\,L\,d_{\text{model}}\,d_{\text{ff}}
\;\Longrightarrow\;
L > \frac{d_{\text{ff}}}{2},
$$

so the answer is $\lceil d_{\text{ff}}/2\rceil$.

## Task

Implement the function `crossover_seq_len` that takes two integers, `d_model` and `d_ff`, and returns the smallest integer sequence length $L$ for which the FLOPs of multi‑head self‑attention exceed those of the feed‑forward network.  The function must return an `int`.

```python
def crossover_seq_len(d_model: int, d_ff: int) -> int:
    ...
```

## Example

```python
>>> crossover_seq_len(768, 3072)
1536
>>> crossover_seq_len(512, 2049)
1025   # ceil(2049/2)
```

## What the gate checks

Two gates are applied:

1. **exact_match** – the returned integer must equal the reference value computed by the grader.
2. **is_int** – the return type must be `int`.

The grader uses NumPy to compute the reference and verifies that your implementation matches it exactly.
