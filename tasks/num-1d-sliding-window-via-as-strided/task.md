## Context

A NumPy array is a *buffer* plus a *shape* plus a *strides* tuple. Element
$(i_0, \dots, i_{k-1})$ lives at byte offset

$$\text{offset}(i) = \sum_{d} i_d \cdot s_d ,$$

where $s_d$ is the stride of dimension $d$. Nothing forces the strides to be the
"natural" C-contiguous ones, and nothing forces different index tuples to map to
*different* offsets. That is the whole trick behind zero-copy sliding windows.

For a 1D array $x$ of length $N$ with element stride $s$, the matrix of all
length-$w$ contiguous windows

$$W_{i,j} = x_{i+j}, \qquad 0 \le i \le N-w, \quad 0 \le j < w$$

has offset $\text{offset}(i,j) = (i+j)\,s = i\,s + j\,s$. So the window matrix is
exactly the same buffer viewed with shape $(N-w+1,\; w)$ and strides $(s,\; s)$ —
no data is copied, and consecutive rows *overlap* in memory.

`np.lib.stride_tricks.as_strided` builds precisely such a view. It performs no
bounds checking: getting the strides wrong reads whatever bytes happen to be
next in memory, so the arithmetic has to be right.

## Task

Implement `sliding_window(x, w)` in `solve.py`.

```python
def sliding_window(x: np.ndarray, w: int) -> np.ndarray:
    ...
```

Requirements:

* `x` is a 1D array of length $N \ge w \ge 1$, of any dtype. It may be
  **non-contiguous** (e.g. `x = base[::3]`), so you must read the real element
  stride from `x` rather than assuming `itemsize`.
* Return an array of shape $(N-w+1,\; w)$ whose row $i$ equals `x[i:i+w]`,
  with the **same dtype** as `x`.
* The result must be a **zero-copy view**: it has to share memory with `x`.
  Materialising the windows (`np.stack`, fancy indexing, list comprehensions,
  `np.copy`) is rejected by the second gate.
* Raise `ValueError` if `w < 1` or `w > N`.

## Example

```python
import numpy as np

x = np.arange(6, dtype=np.int32)
W = sliding_window(x, 3)
print(W)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]
#  [3 4 5]]
print(np.shares_memory(W, x))   # True

base = np.arange(10, dtype=np.float64)
y = base[::2]                   # stride = 16 bytes, not 8
print(sliding_window(y, 2))
# [[0. 2.]
#  [2. 4.]
#  [4. 6.]
#  [6. 8.]]
```

## What the gate checks

The grader builds a reference window matrix for every case with
`np.stack([x[i:i+w] for i in ...])` — a plain, obviously-correct materialisation —
and compares it to your output.

* `byte_exact_fraction` — after verifying shape and dtype match, the raw bytes of
  your result are compared to the reference's. Gate: `>= 1.0`, i.e. every byte of
  every case must be identical.
* `zero_copy_fraction` — fraction of cases where `np.shares_memory(result, x)` is
  true. Gate: `>= 1.0`, so a copied result fails even if its values are perfect.

Both include a case where `x` is a strided slice, and a case where `w == N`.
`ValueError` handling for out-of-range `w` is checked as part of the byte metric.
