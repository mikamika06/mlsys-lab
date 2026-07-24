## Context

CPython frees an object the instant its reference count reaches zero — no
waiting for a garbage-collection pass, as long as the object isn't part of
a reference cycle. That means the *shape of your code*, not the size of
your data, decides how many "large" objects are alive at once: rebinding a
loop variable to a new object immediately drops the refcount on whatever it
pointed to before, and — if nothing else holds a reference — that object's
`__del__` runs right there, synchronously.

Concretely: if a loop **first collects every object into a list** and only
then processes them, all $N$ objects are alive simultaneously (peak live
count $= N$). If it instead processes **one object per iteration** and lets
the loop variable's rebinding drop the previous reference, the peak live
count stays bounded by a small constant, independent of $N$:

$$
\text{peak\_live} = \max_{t} \; \bigl|\{\, \text{objects with refcount} > 0 \text{ at time } t \,\}\bigr|
$$

A straightforward `for` loop that reassigns the same local name each
iteration (`buf = make_buffer(s)`) briefly holds **two** objects live at
the moment of reassignment — CPython creates and binds the new value before
decrementing the old one — so a peak of $2$ is the natural bound for
correct streaming code, not $1$.

## Task

Implement `process`:

```python
def process(sizes: list[int], make_buffer) -> float:
    ...
```

- `sizes` is a list of positive `int`s.
- `make_buffer` is a callable: `make_buffer(size)` constructs and returns a
  large object with a `.checksum() -> float` method. Treat it as an opaque
  large-object factory — call it, don't cache or subclass it.
- For each `size` in `sizes`, construct one large object via
  `make_buffer(size)`, call `.checksum()` on it, and accumulate the sum.
  Return the total.
- **Do not** hold references to more than the minimum necessary number of
  large objects at once — no list of all objects built up front, no
  accumulating list of objects (or of anything that keeps them alive) that
  outlives the iteration that created it.

## Example

```python
def process(sizes, make_buffer):
    total = 0.0
    for s in sizes:
        buf = make_buffer(s)      # previous buf's refcount drops here
        total += buf.checksum()
    return total
```

## What the gate checks

The grader's `make_buffer` constructs real tracked objects: their
`__init__` increments a class-level live-instance counter and updates a
running peak, and their `__del__` decrements it — genuine CPython
reference-counting behaviour, not a simulation. The grader runs your
`process` over several `sizes` lists of different lengths and records the
worst-case peak live count.

- **`peak_ratio`** — `peak_live_count / 2`, taken over the worst of several
  runs, must be `<= 1.0`. Collecting every object into a list first (peak
  $= N$, growing with input size) fails this by a wide and widening margin;
  a genuinely streaming loop stays at peak $\le 2$ regardless of $N$.
- **`total_rel_err`** — the returned total, compared against an
  independently computed reference sum of checksums, must match to a
  relative error `<= 1e-9`. This ensures a low peak count isn't achieved by
  simply not doing the work.
