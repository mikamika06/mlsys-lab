## Context

In a standard multi‑head attention layer the query, key and value matrices are all of shape \((N,d)\).  
The naïve implementation computes the full score matrix \(S = QK^\top\) (size \(N\times N\)), applies a softmax row‑wise, then multiplies by \(V\):
\[
O = \operatorname{softmax}(QK^\top)V .
\]
All intermediate tensors are materialised in HBM.  
FlashAttention processes the computation in blocks of size \(B\).  For each block it loads a tile of \(K\) and \(V\), computes partial scores, applies softmax locally, multiplies by the tile of \(V\), and accumulates into the output.  This reduces the amount of data that must be streamed to/from HBM.

The total number of bytes moved can be counted exactly if we assume each element is a 32‑bit float (4 bytes).  
For the naïve algorithm:

* Read \(Q, K, V\): \(3Nd\) elements
* Write the score matrix: \(N^2\)
* Softmax reads and writes the scores again: \(2N^2\)
* Multiply by \(V\) reads the scores once more and \(V\), then writes the output: \(N^2 + Nd\)

Hence
\[
B_{\text{naïve}} = 5Nd\,4 + 5N^2\,4 .
\]

For FlashAttention with block size \(B\):

* Read \(Q\) once: \(Nd\)
* For each of the \(\lceil N/B\rceil\) blocks:
  * Load a tile of \(K\): \(Bd\)
  * Load a tile of \(V\): \(Bd\)
  * Compute partial scores and write them: \(NB\,4\)
  * Softmax reads and writes the partial scores again: \(2NB\,4\)
  * Multiply by the tile of \(V\) reads the partial scores once more and the tile of \(V\), then writes a block of the output: \(NB + Bd + Nd\)

Thus
\[
B_{\text{flash}} = Nd\,4 \;+\;
\Bigl\lceil \tfrac{N}{B}\Bigr\rceil
\bigl(3NB\,4 + 3Bd\,4 + Nd\,4\bigr).
\]

The metric we ask for is the ratio
\[
R(N,d,B)=\frac{B_{\text{naïve}}}{B_{\text{flash}}}.
\]
Because all terms are integers, \(R\) can be computed exactly in Python using integer arithmetic.

## Task

Implement the function

```python
def hbm_bytes_ratio(N: int, d: int, B: int) -> float:
    """
    Return the ratio of HBM bytes moved by naïve attention to that of FlashAttention.
    N   – number of tokens (rows)
    d   – hidden dimension per token
    B   – block size used by FlashAttention

    The function must use only integer arithmetic and return a Python float.
    """
```

The implementation should follow the formulas described in the context section.

## Example

```python
>>> hbm_bytes_ratio(128, 64, 32)
1.0714285714285714
```

(The exact value depends on the chosen parameters.)

## What the gate checks

A single metric `rel_err` is computed between your result and a reference implementation.  
The relative error must satisfy

\[
\mathrm{rel\_err} \le 10^{-8}.
\]

If this condition holds for all test cases, the task passes.
