## Context

A Python `@property` lets you intercept every attribute write with a setter
that runs real validation before anything is stored:

```python
class C:
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # validate `value` here; raise to reject it -- self._data must not change
        self._data = value
```

If the setter's validation is correct, a **rejected** assignment must be a
complete no-op: the exception propagates and the backing buffer keeps
whatever it held before the attempt. Only an **accepted** assignment may
change the buffer. After a whole scripted sequence of accepted and rejected
writes, the backing buffer's bytes are determined entirely by the *last
accepted* write — nothing about the rejected attempts in between (or after
it) should be visible in the final state.

## Task

Implement:

```python
class ValidatedArray:
    def __init__(self, shape: tuple, dtype=np.float32):
        ...

    @property
    def data(self) -> np.ndarray:
        ...

    @data.setter
    def data(self, value) -> None:
        ...
```

* `__init__` stores `shape` and `dtype` and initializes the backing buffer to
  `np.zeros(shape, dtype=dtype)`.
* The `data` getter returns the current backing buffer.
* The `data` setter accepts a new value **only if** it is an `np.ndarray`
  whose `.shape` equals `self.shape` **exactly** and whose `.dtype` equals
  `self.dtype` **exactly** (no automatic casting or reshaping). On acceptance
  it stores a copy of the array. On any violation — wrong shape, wrong dtype,
  or a value that isn't an `np.ndarray` at all (e.g. a plain list) — it must
  `raise` (any exception is fine) and leave the existing backing buffer
  **completely unchanged**.

## Example

```python
import numpy as np

va = ValidatedArray((3, 4), dtype=np.float32)
good = np.zeros((3, 4), dtype=np.float32)
va.data = good                      # accepted

bad_shape = np.zeros((4, 3), dtype=np.float32)
va.data = bad_shape                 # raises; va.data is still `good`

bad_dtype = np.zeros((3, 4), dtype=np.float64)
va.data = bad_dtype                 # raises; va.data is still `good`
```

## What the gate checks

The grader creates one `ValidatedArray((3, 4), dtype=np.float32)` and drives
it through a fixed scripted sequence of ten assignment attempts — four valid
arrays interleaved with a wrong-shape array, a wrong-dtype (`float64`)
array, a wrong-dtype (`int32`) array, a plain Python list, and (at the very
end, after the last valid write) one more wrong-shape array and one plain
string. It independently tracks, from the script itself, what the buffer's
bytes *should* be after each step (the real oracle: whichever array was the
most recent one that actually satisfies the shape/dtype contract). After
running the whole sequence against your class, it reads `va.data` and
compares it byte-for-byte against that independently-computed expectation
with `byte_exact_fraction` (wrong shape/dtype/length scores `0.0`). The gate
requires `byte_exact_fraction >= 1.0` — the final buffer must match exactly,
which only happens if every accept/reject decision along the way was made
correctly.
