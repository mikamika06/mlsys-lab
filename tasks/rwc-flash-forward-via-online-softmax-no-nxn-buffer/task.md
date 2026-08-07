## Context

Standard scaled dot-product attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

For a sequence of length $N$, this requires materializing an $N \times N$ score matrix — prohibitive for large $N$ due to memory.

**Flash Attention** avoids this by tiling the computation. The key insight is the **online softmax** algorithm: you can compute softmax over a sequence in a single pass by maintaining a running maximum $m$ and a running denominator $\ell$, and rescaling the accumulator whenever the running max is updated.

For each tile of keys/values of block size $B$:

1. Compute scores $s_j = q \cdot k_j / \sqrt{d}$ for keys in the tile.
2. Update running max: $m_{\text{new}} = \max(m, \max_j s_j)$.
3. Rescale accumulator: $\text{acc} \leftarrow \text{acc} \cdot e^{m - m_{\text{new}}}$, $\ell \leftarrow \ell \cdot e^{m - m_{\text{new}}}$.
4. Accumulate: $\text{acc} \leftarrow \text{acc} + \sum_j e^{s_j - m_{\text{new}}} v_j$, $\ell \leftarrow \ell + \sum_j e^{s_j - m_{\text{new}}}$.
5. Update $m \leftarrow m_{\text{new}}$.
6. Final output: $o = \text{acc} / \ell$.

## Task

Implement `flash_attention_forward(Q, K, V, block_size)`:

```python
def flash_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int=32) -> list[list[float]]:
    ...
```

- `Q`, `K`, `V`: 2-D float32 list of shape $(N, d)$.
- `block_size`: tile size for the key/value dimension.
- Returns: float32 array of shape $(N, d)$ — the attention output.

The implementation must process K and V in tiles of `block_size` without ever constructing an $(N, N)$ matrix.

## Example

```python
random.seed(0)
N, d = 4, 8
Q = [[random.gauss(0, 1) for _ in range(d)] for _ in range(N)]
K = [[random.gauss(0, 1) for _ in range(d)] for _ in range(N)]
V = [[random.gauss(0, 1) for _ in range(d)] for _ in range(N)]
out = flash_attention_forward(Q, K, V, block_size=2)
# out shape: (4, 8), matches naive softmax attention within 1e-5
```

## What the gate checks

The grader computes the naive softmax attention output in float64 and compares it to the student's tiled implementation. The **max_abs_err** must be $\le 10^{-5}$ across multiple sequence lengths and block sizes.
