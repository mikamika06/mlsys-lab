## Context

Transformer attention computes $\text{softmax}(QK^\top / \sqrt{d})\,V$ over key positions.
ALiBi (Attention with Linear Biases) replaces learned positional encodings by adding a
deterministic bias to each attention score:

$$\tilde{s}_{ij} = s_{ij} + m \cdot (i - j)$$

where $s_{ij}$ is the raw query-key dot product, $m$ is a pre-chosen slope, $i$ is the
query position and $j$ is the key position. Positions closer to the query receive a
more positive (or less negative) bias.

In memory-efficient attention kernels (e.g.\ FlashAttention), softmax is computed
via the **online algorithm** — one key at a time — so the full $n \times n$ score
matrix is never materialised. The online softmax for a sequence of values
$v_0, v_1, \ldots, v_{n-1}$ maintains a running maximum $M$ and running sum $D$:

$$
M_k = \max(M_{k-1},\; v_k), \qquad
D_k = D_{k-1}\, e^{\,M_{k-1} - M_k} + e^{\,v_k - M_k}
$$

with $M_{-1} = -\infty,\; D_{-1} = 0$. After processing all $n$ values the
softmax output for position $j$ is $e^{\,v_j - M_{n-1}} \,/\, D_{n-1}$.

The challenge: integrate the ALiBi bias *inside* the streaming loop so that
the running max adapts to the shifted scores $v_j = s_{ij} + m(i - j)$ as each
key arrives, without ever forming the full biased matrix first.

## Task

Implement `alibi_online_softmax(scores, slopes)`:

```python
def alibi_online_softmax(scores: list[list[float]], slopes: list[float]) -> list[list[float]]:
    """Return (n, n) softmax probabilities using the online algorithm.

    For each query row i, the effective score at key position j is
        v_j = scores[i, j] + slopes[i] * (i - j).
    Compute softmax over j using the streaming (online) algorithm:
    process keys j = 0, 1, ..., n-1 one at a time, tracking the running
    maximum M and running sum D with the rescaling update rule.
    """
    ...
```

Use Python for array operations but implement the online loop yourself in Python.
The function must return a `float64` array of shape $(n, n)$ whose rows are valid
probability distributions.

## Example

```python
scores = [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]
slopes = [0.25, 0.125, 0.0625]

probs = alibi_online_softmax(scores, slopes)
# Row 0: effective scores = [1+0.25*0, 2+0.25*(-1), 3+0.25*(-2)]
#                        = [1.0, 1.75, 2.5]
# probs[0] ≈ [0.1572, 0.3255, 0.5173]
assert probs.shape == (3, 3)
assert all(abs(sum(row) - 1.0) < 1e-5 for row in probs)
```

## What the gate checks

The gate computes a **Python reference** for each test case: it adds the ALiBi
bias to the full score matrix and applies numerically-stable row-wise softmax.
It then compares the student's output against this reference using
$\text{max\_abs\_err} = \max_{i,j} |\hat{p}_{ij} - p_{ij}|$.
The gate passes when $\text{max\_abs\_err} < 10^{-5}$ across all test cases
(including edge cases with zero slopes, large slopes, and varying matrix sizes).

Five test cases are used: a small $4 \times 4$ matrix, an $8 \times 8$ matrix with
varying slopes, a $3 \times 3$ matrix with large slopes, a zero-score matrix
(testing bias-only softmax), and a $16 \times 16$ matrix with a uniform slope.
All random data uses fixed seeds for determinism.
