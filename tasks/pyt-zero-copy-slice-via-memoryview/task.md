## Context

In CPython, objects that expose the **buffer protocol** (such as `bytes`, `bytearray`, and `array.array`) allow external code to read (or write, if writable) their underlying memory directly without copying. A `memoryview` is a built-in zero-copy view that implements the buffer protocol and exposes a Python buffer interface to arbitrary memory.

Given a contiguous buffer `buf` (e.g. a `bytearray`) and start/stop indices, slicing via `buf[start:stop]` creates a new `bytes` *copy* of the selected range. Using `memoryview`, however, you can return a view that shares the original buffer — no allocation or copy is performed beyond the tiny view object itself.

The identity of the backing object is accessible via the `.obj` attribute of a memoryview: `mv.obj` is the original buffer object (or `None` for some views). Verifying `mv.obj is buf` proves zero-copy.

## Task

Implement `zero_copy_slice(buf, start, stop)`:

```python
def zero_copy_slice(buf: bytearray, start: int, stop: int) -> memoryview:
    ...
```

It must return a `memoryview` that covers the byte range `start` (inclusive) to `stop` (exclusive) of the `bytearray` `buf`, **sharing the same underlying buffer** (no copy). The returned view must satisfy:

1. `bytes(mv) == buf[start:stop]` (byte-exact content).
2. `mv.obj is buf` (the view is backed by the original buffer, not a copy).

Assume `0 <= start < stop <= len(buf)`.

## Example

```python
data = bytearray(b'hello world')
mv = zero_copy_slice(data, 0, 5)
print(bytes(mv))   # b'hello'
print(mv.obj is data)  # True

mv2 = zero_copy_slice(data, 6, 11)
print(bytes(mv2))  # b'world'
print(mv2.obj is data)  # True
```

## What the gate checks

A single gate: `byte_exact_fraction` must equal 1.0. The grader:

1. Constructs a random `bytearray` of length 256.
2. Calls `zero_copy_slice` on a random `[start, stop)` interval.
3. Asserts `bytes(mv) == buf[start:stop]` and `mv.obj is buf`.
4. If either fails, the byte fraction is 0.0; otherwise 1.0.

A naive `return bytes(buf[start:stop])` returns a `bytes` object (not a memoryview) and creates a copy — it will fail because `bytes` has no `.obj` attribute (or `mv.obj` will be different). A wrong implementation that uses `memoryview(buf[start:stop])` also fails because `mv.obj` will be a temporary `bytes` copy, not the original buffer.
