## Context

The buffer protocol lets a Python object expose its raw memory to other code
without copying it. `memoryview(obj)` wraps any object that implements the
protocol — including a NumPy array — and reports properties of the
*underlying buffer*, independent of any NumPy-specific bookkeeping. One of
those properties is `memoryview.c_contiguous`: whether the buffer's bytes are
laid out so that stepping through the last axis fastest, then the next axis,
and so on, visits memory in strictly increasing address order with no gaps.

For an array of shape $(d_0, \dots, d_{n-1})$ and item size $s$, the
canonical C-contiguous strides are

$$
\mathrm{stride}_k = s \prod_{j=k+1}^{n-1} d_j .
$$

Slicing with a non-unit step, transposing, or reversing an axis changes the
strides away from this canonical pattern (often introducing negative or
enlarged strides), so the resulting view is no longer C-contiguous even
though it still points into the same underlying buffer. Some operations
(like `reshape` to a flat shape) cannot always be expressed as a
strided view of non-contiguous data, so NumPy silently falls back to
allocating a fresh, contiguous buffer — the result is contiguous again,
regardless of what came before it.

## Task

Implement `predict_c_contiguous(ops)`:

```python
def predict_c_contiguous(ops: list[str]) -> list[bool]:
    ...
```

The input is a sequence of operation names. Start from the base array

```python
np.arange(24, dtype=np.int64).reshape(4, 6)
```

and apply each operation in order, carrying the result of one operation into
the next. Return a list of booleans, one per operation, where each entry is
`True` if the array *after that operation* is C-contiguous according to the
buffer protocol (`memoryview(array).c_contiguous`), and `False` otherwise.

Supported operations:

- `"transpose"`: apply `array.T`.
- `"slice_step2"`: apply `array[::2]`.
- `"reshape_flat"`: apply `array.reshape(-1)`.
- `"flip"`: apply `array[::-1]`.

## Example

```python
ops = ["transpose", "reshape_flat"]
predict_c_contiguous(ops)
# [False, True]
```

Transposing a 2-D array swaps its strides away from the canonical C order, so
the transposed view is not C-contiguous. `reshape(-1)` on that non-contiguous
data cannot be expressed as a view, so NumPy copies into a fresh contiguous
buffer — the flattened result is contiguous again.

## What the gate checks

The grader replays several operation sequences (including ones that revisit
contiguity, like `["transpose", "transpose"]`, and ones that recover
contiguity through a copy, like `["flip", "reshape_flat"]`) against the real
base array and records `memoryview(array).c_contiguous` after each step as
the reference. Your returned list must equal that reference exactly on every
sequence for `exact_match` to be $1.0$ — a classifier that assumes a given
operation always has the same contiguity effect regardless of what came
before it will get at least one sequence wrong.
