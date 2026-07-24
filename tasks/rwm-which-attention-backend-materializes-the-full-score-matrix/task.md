## Context

In transformer‑style attention, the core operation is a matrix multiplication between queries $Q \in \mathbb{R}^{N\times d}$ and keys $K^\top \in \mathbb{R}^{d\times N}$.  The result is an $N\times N$ *score* matrix that is then scaled, masked, and soft‑maxed to produce attention weights.  
The naive implementation materialises this full $O(N^2)$ score matrix in memory before applying the soft‑max.  More recent backends avoid this by computing soft‑max on the fly or using block‑wise strategies; they are called *memory‑efficient*.

Four representative backends appear in practice:

| Backend | Memory behaviour |
|---------|------------------|
| `naive` | materialises full $N\times N$ score matrix |
| `math-SDPA` | uses the “scaled dot‑product attention” routine that keeps the full matrix |
| `mem-efficient` | computes soft‑max without storing all scores |
| `flash` | a highly optimised kernel that never stores the full matrix |

The task is to classify a backend by whether it materialises the full score matrix.

## Task

Implement the function:

```python
def classify_backend(name: str) -> str:
    ...
```

It receives the name of an attention backend (one of `"naive"`, `"math-SDPA"`, `"mem-efficient"`, `"flash"`).  
Return a string label that indicates whether the backend materialises the full $N\times N$ score matrix:

* `"full"` – if it does materialise the matrix
* `"efficient"` – otherwise

The function must be pure (no side effects) and run in constant time.

## Example

```python
>>> classify_backend("naive")
'full'
>>> classify_backend("flash")
'efficient'
```

## What the gate checks

A single gate named `exact_match` compares your output against an oracle that knows the correct classification for each backend name.  The grader calls your function on all four names and requires exact equality of the returned labels.
