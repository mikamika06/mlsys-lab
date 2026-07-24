## Context

Python's cyclic garbage collector can reclaim groups of objects that only reference
each other. A problem appears when object finalization is implemented with
`__del__` methods because the finalizer becomes part of the object's lifecycle and
can complicate predictable cycle cleanup.

The `weakref.finalize` helper separates cleanup registration from object lifetime.
The callback is stored by the weak reference machinery, allowing the cyclic garbage
collector to reclaim the object graph while still executing the cleanup action.

Consider a cycle of objects $a$ and $b$ where

$$a.\mathrm{next}=b,\qquad b.\mathrm{next}=a.$$

The cycle has no external references after both variables are cleared. The garbage
collector should be able to find and collect this strongly connected component while
the registered finalizer records that cleanup occurred.

## Task

Implement `collect_cycle_with_finalizer()`.

The function must:

1. Create two objects that reference each other.
2. Register a cleanup callback using `weakref.finalize`, not `__del__`.
3. Remove all strong references to the objects.
4. Force garbage collection.
5. Return a list describing the observed cleanup events.

The returned list must contain the callback event followed by the finalizer state:

```python
["finalized", "dead"]
```

The implementation should not depend on object addresses, timing, or interpreter
shutdown behavior.

## Example

```python
result = collect_cycle_with_finalizer()
print(result)
# ["finalized", "dead"]
```

## What the gate checks

The gate builds the expected result by running an independent implementation that
uses CPython's actual `weakref.finalize` behavior and garbage collector. The returned
event sequence from `collect_cycle_with_finalizer()` must exactly match that oracle.

A solution that relies on `__del__` finalizers instead of `weakref.finalize` will not
produce the required cleanup event sequence.
