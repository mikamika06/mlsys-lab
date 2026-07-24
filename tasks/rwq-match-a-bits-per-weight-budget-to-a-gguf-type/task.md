## Context

In many quantised neural‑network formats the *bits per weight* (bpw) is a key design parameter.  
A GGUF type is defined by a block layout: each block occupies a fixed number of bytes and contains a fixed number of weights.  
If a block has $B$ bytes and holds $W$ weights, then

$$
\text{bpw} = \frac{8\,B}{W}\;.
$$

Typical GGUF types used in practice are:

| Type      | Block size (bytes) | Weights per block |
|-----------|--------------------|-------------------|
| Q2_K      | 16                 | 64                |
| Q4_K_M    | 32                 | 64                |
| Q6_K      | 48                 | 64                |
| Q8_0      | 64                 | 64                |

These give bpw values of $2,\,4,\,6,$ and $8$ respectively.

## Task

Implement `match_bpw(target_bpw: float) -> int`:

```python
def match_bpw(target_bpw: float) -> int:
    ...
```

The function receives a target bits‑per‑weight budget and must return the **index** (0‑based) of the GGUF type whose computed bpw is numerically closest to `target_bpw`.  
If two types are equally close, choose the one with the smaller index.  
Return values:

- 0 → Q2_K
- 1 → Q4_K_M
- 2 → Q6_K
- 3 → Q8_0

The implementation must use only NumPy for numeric operations and be deterministic.

## Example

```python
>>> match_bpw(5.3)
2          # closest to 6 (Q6_K)

>>> match_bpw(7.9)
3          # closest to 8 (Q8_0)

>>> match_bpw(1.8)
0          # closest to 2 (Q2_K)
```

## What the gate checks

The grader computes the exact bpw values for all four types using NumPy, finds the index of the type with minimal absolute difference to the supplied target, and compares it to the candidate’s output.  
A correct implementation must return exactly that index for every test case.
