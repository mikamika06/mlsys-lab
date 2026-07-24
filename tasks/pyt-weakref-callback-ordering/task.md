## Context

Python's `weakref` module provides weak references to objects. When an object has only weak references pointing to it, it can be garbage collected. If a callback is registered with a weak reference via `weakref.ref(obj, callback)`, that callback fires during garbage collection with the weak reference as its sole argument. When multiple objects in reference cycles are collected by the cyclic garbage collector, the order of finalization (and thus callback invocation) is generally the reverse of the order in which objects were created. This is a CPython implementation detail (not a language guarantee), but for deterministic measurement we can rely on it within a single Python session.

The weakref callback receives the *dead* reference as its first argument. By storing the referent's identity or a name before it dies, we can record the order in which callbacks fire. The cyclic GC processes objects in the order they are added to the garbage list, which typically corresponds to creation order during a `gc.collect()` call after all strong references are dropped.

## Task

Implement `get_callback_order(objects)`:

```python
def get_callback_order(objects: list[tuple[str, object]]) -> list[str]:
    """
    Returns the order in which weak reference callbacks fire.

    Args:
        objects: List of (name, obj) pairs. Each obj is an object that supports
                 weak references (e.g., an instance of a class). The name is a
                 string identifier for recording callback order.

    Returns:
        A list of strings (the names) in the order their weakref callbacks
        were invoked during garbage collection.
    """
```

Your function should:

1. Register a weak reference callback for each object that, when fired, appends the associated name to an internal list.
2. Ensure all strong references to the objects are dropped (except the weak references themselves).
3. Force garbage collection with `gc.collect()` to trigger callbacks.
4. Return the list of names in the order callbacks were called.

**Important:** The test harness will create objects in a specific order and expect callbacks to fire in the reverse of creation order (the standard CPython cyclic GC behavior for a single generation). Your implementation must not rely on external state or global lists that persist across calls; it must compute the order fresh each time.

## Example

```python
import weakref

class Dummy:
    pass

a, b, c = Dummy(), Dummy(), Dummy()
objects = [("first", a), ("second", b), ("third", c)]

order = get_callback_order(objects)
print(order)  # Expected: ['third', 'second', 'first']
```

In this example, `a` was created first, `c` last. After dropping all strong references and calling `gc.collect()`, the cyclic GC collects them in reverse creation order, so the callback for `c` (name "third") fires first, then `b`, then `a`.

## What the gate checks

A single exact‑match gate: the list of names returned by `get_callback_order(objects)` must exactly equal the reference order computed by the check.py grader. The grader constructs its own objects in a deterministic creation order, calls your function, and compares the result with the expected reverse‑creation order. If your function returns a different order or crashes, the gate fails.
