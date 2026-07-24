## Context

Every CPython object begins with a common object header. One field in that
header stores the reference count. For ordinary objects, the reference count
changes as references are created and destroyed.

CPython 3.12 introduced immortal objects. Some objects that are always present
or widely shared use a special reference count sentinel instead of a normal
small reference count. Conceptually, the header contains a value

$$\text{ob\_refcnt} =
\begin{cases}
S & \text{for immortal objects} \\
N & \text{for ordinary objects}
\end{cases}$$

where $S$ is the immortal sentinel value and $N$ is the current reference
count.

The Python `id()` of an object points at the start of its CPython object memory,
so the reference count field can be inspected through the object header layout.

## Task

Implement `immortal_refcount_sentinel_detector()`:

```python
def immortal_refcount_sentinel_detector() -> list[bool]:
    ...
```

The function must inspect this fixture set:

- `None`
- `True`
- `False`
- the small integers from `-5` through `256`
- several interned strings
- one fresh `object()` instance

Return a boolean list with one entry per fixture. Each entry is `True` when
that object has the immortal reference count sentinel and `False` when it has a
normal reference count.

Use CPython object-header inspection. Do not use object identity shortcuts
such as `obj is None` or type checks to classify the objects.

## Example

```python
flags = immortal_refcount_sentinel_detector()

# The result shape is a boolean vector matching the internal fixture order.
# The exact values depend on the CPython build, but CPython 3.12+ fixtures
# include both immortal and ordinary objects.
print(len(flags))
# 268+
```

## What the gate checks

The gate builds the same fixture collection and computes the expected boolean
vector by reading the real CPython object header reference counts at grading
time.

The returned list must exactly match the oracle classification. This checks that
the implementation detects the actual immortal reference count sentinel rather
than relying on assumptions about object types or values.
