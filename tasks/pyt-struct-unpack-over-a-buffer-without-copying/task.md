## Context

`struct.unpack` and `numpy.frombuffer` both decode raw bytes, but they answer
different questions. `struct.unpack` always *copies*: it allocates brand-new
Python `int`/`float` objects from the bytes it reads. A `memoryview`, by
contrast, implements the buffer protocol and can be **sliced** and **cast**
to a different element format without touching a single byte of the
underlying storage — `memoryview(buf)[a:b].cast(fmt)` is a *view*, not a
copy: mutate the original buffer afterwards and the view sees the change
immediately, because it is still looking at the exact same memory.

A packed record buffer holds $n$ fixed-size records back to back. Here each
record is 12 bytes, little-endian `<iff>`: one `int32` id, then two `float32`
fields $x, y$.

$$
\text{record}_i \text{ occupies bytes } [12i,\ 12i+12), \quad
\text{id} = \text{bytes } [12i, 12i{+}4),\ \
x = [12i{+}4, 12i{+}8),\ \
y = [12i{+}8, 12i{+}12).
$$

## Task

Implement `parse_records_view`:

```python
def parse_records_view(buf: bytearray, n: int, record_size: int = 12) -> list:
    ...
```

For each of the `n` records, build **three zero-copy `memoryview` objects**
`(id_view, x_view, y_view)` via slicing `memoryview(buf)` and `.cast(...)` —
`id_view` cast to format `'i'` (int32), `x_view`/`y_view` cast to format
`'f'` (float32). Do **not** use `struct.unpack`, `numpy.frombuffer`, or
`bytes(...)` anywhere — any operation that copies bytes defeats the point.
Return a list of `n` tuples `(id_view, x_view, y_view)`.

## Example

```python
import struct
buf = bytearray(struct.pack('<iff', 7, 1.5, -2.5))
views = parse_records_view(buf, 1)
id_view, x_view, y_view = views[0]
id_view[0], x_view[0], y_view[0]      # (7, 1.5, -2.5)

buf[0:4] = struct.pack('<i', 99)      # mutate the ORIGINAL buffer
id_view[0]                            # -> 99  (the view sees the live buffer)
```

## What the gate checks

* `exact_match` — the grader builds a buffer of 40 records from a real
  `numpy` reference (fresh random `int32`/`float32` values, seeded), calls
  your function, checks every returned element **is a `memoryview`** (not a
  plain number — a `struct.unpack`-based shortcut fails this), and compares
  every `id_view[0]`/`x_view[0]`/`y_view[0]` to the reference values. Gate:
  `== 1.0`.
* `zero_copy_fraction` — after the initial parse, the grader mutates several
  bytes of the **original** `buf` in place (new id/x/y values at a few record
  positions) and re-reads the *already-returned* views: a true zero-copy view
  must reflect the new bytes immediately, with no further calls into your
  function. Gate: `>= 1.0`.
