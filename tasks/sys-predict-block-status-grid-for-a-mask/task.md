## Context

FlashAttention and similar fused-attention kernels partition the $n \times n$ 
attention matrix into a grid of rectangular tiles of size $s_h \times s_w$. 
Before issuing the inner product for a tile the kernel inspects the associated 
key/query masks and classifies the tile into one of three categories:

$$\text{status}_{pq} = 
\begin{cases}
0 \;\text{(masked)}   & \text{if every cell in tile } (p,q) \text{ is masked}, \\[4pt]
1 \;\text{(partial)}  & \text{if some cells are masked and some are not}, \\[4pt]
2 \;\text{(dense)}    & \text{if no cell in tile } (p,q) \text{ is masked}.
\end{cases}$$

Tiles classified as **masked** ($0$) are skipped entirely — their contribution to 
the softmax numerator is zero. **Dense** tiles ($2$) enter the fast path that 
assumes every element participates. Only **partial** tiles ($1$) need the 
slow element-wise path with per-element masking and boundary handling.

Concretely, given a boolean mask $\mathbf{M} \in \{0,1\}^{H \times W}$ (where 
$1$ = masked) and tile dimensions $s_h, s_w$ that evenly divide $H$ and $W$, 
the status for tile $(p, q)$ covering rows 
$\bigl[p\,s_h,\;(p+1)\,s_h\bigr)$ and columns 
$\bigl[q\,s_w,\;(q+1)\,s_w\bigr)$ is determined by the tile sum:

$$S_{pq} = \sum_{i=p\,s_h}^{(p+1)\,s_h - 1} \;\sum_{j=q\,s_w}^{(q+1)\,s_w - 1} \mathbf{M}_{ij}$$

The classification then follows directly:

$$\text{status}_{pq} =
\begin{cases}
0 & \text{if } S_{pq} = s_h \cdot s_w \;\;(\text{all masked}), \\[4pt]
2 & \text{if } S_{pq} = 0 \;\;(\text{all unmasked}), \\[4pt]
1 & \text{otherwise}.
\end{cases}$$

## Task

Implement `block_status_grid(mask, tile_h, tile_w)`:

```python
import numpy as np

def block_status_grid(mask: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    ...
```

**Parameters:**

- `mask`: a boolean 2-D NumPy array of shape $(H, W)$. `True` means the cell is masked.
- `tile_h`, `tile_w`: positive integers that evenly divide $H$ and $W$ respectively.

**Returns:** an `int32` NumPy array of shape $\bigl(H / s_h,\; W / s_w\bigr)$ where 
entry $(p, q)$ is $0$ (fully masked), $1$ (partially masked), or $2$ (fully 
unmasked / dense).

Use vectorized NumPy operations — no Python `for` loops over individual tiles.

## Example

```python
import numpy as np

mask = np.array([
    [True,  True,  False, False],
    [True,  False, False, False],
    [False, False, False, True ],
    [False, False, True,  True ],
], dtype=bool)

S = block_status_grid(mask, 2, 2)
# Tile (0,0): rows 0-1, cols 0-1 → [T,T; T,F] → sum=3 → partial → 1
# Tile (0,1): rows 0-1, cols 2-3 → [F,F; F,F] → sum=0 → dense   → 2
# Tile (1,0): rows 2-3, cols 0-1 → [F,F; F,F] → sum=0 → dense   → 2
# Tile (1,1): rows 2-3, cols 2-3 → [F,T; T,T] → sum=3 → partial → 1
# array([[1, 2],
#        [2, 1]], dtype=int32)
```

## What the gate checks

The gate builds eight deterministic test cases using a seeded NumPy RNG 
(`np.random.RandomState(42)`). The cases include: all-masked, all-unmasked, 
random 50 % masks with square and rectangular tiles, 1×1 tiles, a 
non-square mask, and an upper-triangular mask. For each case the grader 
computes the ground-truth status matrix with a NumPy reshape-and-sum oracle 
and checks **byte-exact** agreement (`np.array_equal`) with your output. 
Any wrong value, wrong shape, or exception causes failure.
