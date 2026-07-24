## Context

In Triton (a Python-to-GPU DSL), a kernel launches over a 1-D grid of
*blocks*. Each block is assigned a unique thread-block ID via
`tl.program_id(axis=0)` and processes a contiguous tile of $B$ elements
(the *block size*). Given input vectors $a, b \in \mathbb{R}^N$, block
$p$ (for $p = 0, 1, \dots, \lceil N/B \rceil - 1$) is responsible for
indices

$$S_p = \{\, pB,\; pB+1,\; \dots,\; \min(pB + B - 1,\; N-1) \,\} .$$

When $B$ does not divide $N$ evenly, the last block covers fewer than $B$
elements. On real hardware, loading past the end of an allocation is
undefined behavior, so every block constructs a *boundary mask*
$m \in \{0,1\}^B$ with

$$m_j = \begin{cases} 1 & \text{if } pB + j < N, \\ 0 & \text{otherwise,} \end{cases}$$

and applies `tl.where(m, loaded, 0)` before computing. The result is the
same as a plain element-wise sum $c_i = a_i + b_i$, but the program
exercises the blocked + masked lowering pattern that real Triton kernels
rely on.

## Task

Implement `emulated_triton_add`:

```python
import numpy as np

def emulated_triton_add(a: np.ndarray,
                        b: np.ndarray,
                        block_size: int) -> np.ndarray:
    ...
```

**Inputs.** `a` and `b` are 1-D NumPy arrays of the same length $N$ (may
be zero-length). `block_size` is a positive integer $B$.

**Behavior.** Emulate a Triton kernel launch: compute the grid size as
$\lceil N / B \rceil$ blocks, iterate over block IDs $p = 0, \dots,
\lceil N/B \rceil - 1$, build the boundary mask $m$ for each block, mask
the loaded tiles of `a` and `b` (filling out-of-range lanes with 0),
add the masked tiles, and write the valid portion of the result into the
output array.

**Output.** Return a 1-D NumPy array of length $N$ containing
$c_i = a_i + b_i$ for every valid index. The dtype must match
`a.dtype`.

## Example

```python
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
c = emulated_triton_add(a, b, block_size=2)
# Grid: ceil(5/2) = 3 blocks
# Block 0 (pid=0): indices 0,1 — mask [1,1] → [1+5, 2+4] = [6, 6]
# Block 1 (pid=1): indices 2,3 — mask [1,1] → [3+3, 4+2] = [6, 6]
# Block 2 (pid=2): index  4   — mask [1,0] → [5+1, 0]   = [6, 0]
# Valid writes: output = [6, 6, 6, 6, 6]
```

## What the gate checks

The gate metric is `max_abs_err`. It compares your output against the
NumPy oracle $c = a + b$ across ten test cases:

- $N$ exactly divisible by $B$ (no partial block).
- $N$ not divisible by $B$ (partial final block, the masking case).
- $N < B$ (a single block that is wider than the input).
- $N = 0$ (empty input, zero blocks).
- Large random arrays ($N = 100$, $N = 1000$).

A correct implementation returns `max_abs_err == 0`. Failing to mask
the last block, using floor division for the grid size, or ignoring
the boundary all produce nonzero errors on the partial-block cases.
