## Context

Layer Normalisation (LayerNorm) normalises each sample in a batch independently by subtracting the mean and dividing by the standard deviation of its features.  
For an input matrix $X \in \mathbb{R}^{N\times D}$, the forward pass requires computing

$$\mu_i = \frac{1}{D}\sum_{j=1}^D X_{ij}, \qquad
\sigma_i^2 = \frac{1}{D}\sum_{j=1}^D (X_{ij}-\mu_i)^2,$$

for every row $i$.  
A naïve implementation performs two separate passes over the data: one to accumulate the sum for $\mu$, and a second to accumulate squared deviations for $\sigma^2$.  Each element of $X$ is therefore read twice, yielding $2ND$ memory reads.

An optimised single‑pass algorithm can compute both the mean and variance in one traversal by accumulating the sum and the sum of squares simultaneously.  This reduces the number of memory reads to a single pass over each element, i.e. $ND$ reads.

The task is to quantify these two approaches for arbitrary batch size $N$ and feature dimension $D$.

## Task

Implement `count_memory_passes(N: int, D: int) -> Tuple[int, int]`:

```python
def count_memory_passes(N: int, D: int) -> tuple[int, int]:
    ...
```

The function must return a pair `(two_pass_reads, single_pass_reads)` where

- `two_pass_reads` is the total number of element reads performed by the naïve two‑pass algorithm,
- `single_pass_reads` is the total number of element reads performed by an optimised single‑pass algorithm.

Both counts should be exact integers.  The implementation must use only standard Python and NumPy; no external profiling libraries are allowed.

## Example

```python
>>> count_memory_passes(3, 4)
(24, 12)

# Explanation:
# two-pass: 2 * 3 * 4 = 24 reads
# single-pass: 1 * 3 * 4 = 12 reads
```

## What the gate checks

The grader evaluates the function on a set of held‑out $(N,D)$ pairs and verifies that the returned counts match the mathematically exact values.  The comparison is an exact integer equality test; any deviation causes the gate to fail.
