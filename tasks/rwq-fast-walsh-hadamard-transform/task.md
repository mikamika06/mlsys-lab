## Context

Random Hadamard rotations are the workhorse behind outlier-smoothing
quantization methods like QuaRot and SpinQuant: before quantizing
activations or weights, the tensor is multiplied by an orthogonal
Hadamard matrix so that large per-channel outliers get spread evenly
across all channels, shrinking the dynamic range each quantization bin
has to cover. Doing this rotation as a dense $n \times n$ matrix multiply
costs $O(n^2)$. The **Fast Walsh-Hadamard Transform (FWHT)** computes the
exact same rotation in $O(n \log n)$ via a butterfly network, the same
trick FFT uses for the DFT — which is why production kernels (e.g. the
`fast-hadamard-transform` CUDA kernel used by QuaRot/SpinQuant) always use
the butterfly, never the dense matrix.

### Definition

The unnormalized Hadamard matrix is defined recursively:
$$
H_1 = \begin{bmatrix}1\end{bmatrix}, \qquad
H_{2m} = \begin{bmatrix} H_m & H_m \\ H_m & -H_m \end{bmatrix}.
$$

For a vector $x \in \mathbb{R}^n$ with $n = 2^k$, the **normalized**
Hadamard transform is
$$
y = \frac{1}{\sqrt{n}} H_n\, x,
$$
which is orthogonal ($H_n^\top H_n = n I$, so $\frac{1}{\sqrt n}H_n$ is its
own inverse).

### Butterfly algorithm

Instead of building $H_n$, compute $y$ in $\log_2 n$ passes. Starting with
$y \leftarrow x$ and $h = 1$: while $h < n$, for every block of $2h$
consecutive entries, split it into its first half $a$ (indices
$0,\dots,h-1$ within the block) and second half $b$ (indices
$h,\dots,2h-1$), and replace the block with $[a+b,\ a-b]$; then double
$h$. After $\log_2 n$ passes, divide the whole vector by $\sqrt{n}$.

## Task

Implement:

```python
def fwht(x: np.ndarray) -> np.ndarray:
    ...
```

* `x` — 1-D array of length $n = 2^k$ for some integer $k \ge 0$.
* Returns the normalized transform $y = \frac{1}{\sqrt n} H_n x$ as a
  length-$n$ array, computed via the $O(n \log n)$ butterfly above (not a
  dense matrix multiply).

## Example

```python
import numpy as np

x = np.array([1.0, 0.0, 1.0, 0.0])
y = fwht(x)
# H_4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
# H_4 @ x = [2, 2, 0, 0]  ->  y = [2,2,0,0] / sqrt(4) = [1, 1, 0, 0]
```

## What the gate checks

* **max_abs_err** — on several random vectors of length $8, 32, 128, 256$
  (fixed seeds), the maximum absolute difference between your `fwht(x)`
  and the oracle $\frac{1}{\sqrt n} H_n x$ built by explicitly constructing
  $H_n$ from the recursive block rule above and multiplying densely; must
  be $\le 10^{-6}$.
