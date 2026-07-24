## Context

CPython uses reference counting to track how many active references point to an
object. The built-in function `sys.getrefcount(x)` exposes this count, but it
also receives `x` as an argument. That argument creates one temporary reference
during the call.

If the real number of references to an object is $r$, the value returned by
`sys.getrefcount(x)` while evaluating the call is:

$$
\operatorname{getrefcount}(x) = r + 1 .
$$

A function that accepts `x` as an argument has another reference from its local
parameter. Therefore, when implementing a helper around `sys.getrefcount`,
the temporary references introduced by the helper itself must be removed.

## Task

Implement `true_refcount(x)`:

```python
def true_refcount(x):
    ...
```

The function must return the reference count that the object has from the
caller's perspective. It should use `sys.getrefcount` and correct for the
temporary references introduced while measuring.

## Example

```python
import sys

def show(x):
    return true_refcount(x)

obj = []
count = show(obj)

# count is the number of references to obj outside the measurement call
```

The exact numeric value depends on the references held at runtime, so the
function must derive the value dynamically.

## What the gate checks

The gate creates real CPython objects and compares the returned value against an
oracle computed with CPython's own `sys.getrefcount` behavior. The result must
match exactly for every tested object.
