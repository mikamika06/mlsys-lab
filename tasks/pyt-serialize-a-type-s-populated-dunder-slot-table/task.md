## Context

Python type objects expose behavior through a combination of ordinary attributes and
C-level slots. Many dunder names correspond to entries that a type object may
populate. For a class $T$, we can represent the populated dunder table as a binary
vector $m \in \{0,1\}^{40}$:

$$
m_i =
\begin{cases}
1 & \text{if dunder } i \text{ is present on } T,\\
0 & \text{otherwise.}
\end{cases}
$$

A compact serialization can store this bit vector as a bit mask. If bit $i$ is
assigned to the $i$-th name in a fixed ordered list, the mask is

$$
M = \sum_{i=0}^{39} m_i 2^i .
$$

The mask can be stored as an unsigned 64-bit little-endian integer. A sequence of
class masks can therefore be serialized as consecutive 8-byte chunks.

This task uses Python's own type machinery as the source of truth. The populated
status is determined by checking whether the dunder can be retrieved from the type
object with `getattr`, which reflects inherited methods and C-provided behavior.

## Task

Implement `serialize_dunder_slots(classes)`:

```python
def serialize_dunder_slots(classes: list[type]) -> bytes:
    ...
```

The function receives a list of Python classes and returns bytes containing one
little-endian unsigned 64-bit mask per class.

Use this fixed ordered dunder list:

```python
DUUNDERS = [
    "__new__", "__init__", "__repr__", "__str__", "__bytes__",
    "__format__", "__lt__", "__le__", "__eq__", "__ne__",
    "__gt__", "__ge__", "__hash__", "__bool__", "__len__",
    "__iter__", "__next__", "__getitem__", "__setitem__", "__delitem__",
    "__contains__", "__call__", "__enter__", "__exit__", "__await__",
    "__aiter__", "__anext__", "__add__", "__sub__", "__mul__",
    "__matmul__", "__truediv__", "__floordiv__", "__mod__",
    "__pow__", "__neg__", "__pos__", "__abs__", "__invert__",
    "__index__"
]
```

For every class, bit $i$ must be set when `getattr(cls, DUUNDERS[i], None)`
returns a non-`None` value. Return the packed masks in the same order as the input
classes.

## Example

```python
class A:
    def __len__(self):
        return 3

class B:
    def __add__(self, other):
        return other

result = serialize_dunder_slots([A, B])
```

The result contains two 8-byte integers. The first integer has the bit for
`"__len__"` set, and the second integer has the bit for `"__add__"` set.

## What the gate checks

The gate creates ten classes with different combinations of Python-defined and
inherited dunder behavior. It computes the expected masks using the real CPython
`getattr` behavior at grading time, then compares the returned bytes to that
oracle result.

The `byte_exact_fraction` score must equal $1.0$, meaning every output byte must
match the reference serialization.
