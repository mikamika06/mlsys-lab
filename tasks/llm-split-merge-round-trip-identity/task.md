## Context

In multi‑head attention the query, key and value tensors are split into several independent “heads”.  
Let $X \in \mathbb{R}^{B\times T\times D}$ be a batch of sequences with hidden dimension $D$.  
If we choose $H$ heads, each head has size $d = D/H$ (assume $D$ is divisible by $H$).  
The split operation rearranges the last axis so that

$$
X_{\text{split}} \in \mathbb{R}^{B\times T\times H\times d},
\qquad
X_{\text{split}}[b,t,h,:] = X[b,t, h\,d : (h+1)d].
$$

The merge operation performs the inverse permutation and restores the original shape.

## Task

Implement two functions:

```python
def split_heads(x: list[list[list[float]]], num_heads: int) -> list[list[list[list[float]]]]:
    ...

def merge_heads(heads: list[float]) -> list[float]:
    ...
```

`split_heads` must take a 3‑D array `x` of shape `(B,T,D)` and an integer `num_heads` that divides `D`.  
It should return an array of shape `(B,T,num_heads,D//num_heads)`.

`merge_heads` takes the output of `split_heads` and returns a tensor of shape `(B,T,D)` that is identical to the input of `split_heads`.

Both functions must use only Python operations; no explicit Python loops are allowed.

## Example

```python
x = list(range(24)).reshape(2,3,4)   # B=2, T=3, D=4
heads = split_heads(x, 2)
# heads.shape == (2,3,2,2)
merged = merge_heads(heads)
assert merged.shape == (2,3,4)
assert merged == x
```

## What the gate checks

The grader computes a reference implementation using Python and compares your `merge_heads(split_heads(x))` against it.  
It reports the maximum absolute difference:

$$
\max_{i,j,k} |\, \text{candidate}[i,j,k] - \text{reference}[i,j,k] \,|.
$$

The solution must achieve a value $\le 10^{-7}$.
