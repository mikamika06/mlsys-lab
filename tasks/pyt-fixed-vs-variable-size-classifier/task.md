## Context

In CPython every object is represented by a C structure that starts with a universal header containing a reference counter and a pointer to the type object.  
For objects whose size depends on their contents (lists, dictionaries, strings, bytes, tuples, sets, …) this header is followed by a *variable‑size* field `ob_size`.  Such types are called **PyVarObject**s.  
Fixed‑size types (int, float, bool, NoneType, complex, etc.) have no `ob_size` and their memory footprint does not change when the value changes.

The Python runtime exposes the size of an object in bytes through ``sys.getsizeof``.  For a PyVarObject the returned value grows with the number of elements or characters it holds; for fixed‑size types it is constant.

## Task

Implement `classify_objects(objs)`:

```python
def classify_objects(objs: Iterable[Any]) -> np.ndarray:
    ...
```

It receives an iterable of arbitrary Python objects and returns a NumPy array of shape `(len(objs),)` with dtype ``bool``.  
Each element is ``True`` if the corresponding object is a variable‑size PyVarObject whose size depends on its contents, otherwise ``False``.

The implementation must be deterministic across CPython runs; use only standard library modules.

## Example

```python
import numpy as np

objs = [[], {}, set(), (), "", b"", bytearray(),
        0, 1.5, True, None,
        [1,2], {"a":3}, "abc"]

mask = classify_objects(objs)
print(mask)
# array([ True,  True,  True,  True,  True,  True,  True,
#         False, False, False, False,  True,  True,  True])
```

## What the gate checks

The grader builds a fixed list of thirty diverse objects (lists, dicts, tuples, strings, bytes, integers, floats, booleans, ``None``, complex numbers, memoryviews, etc.).  
For each object it computes the reference answer by comparing its size to that of an empty instance of the same type.  The candidate’s output must match this reference exactly; otherwise the gate fails.

The metric used is **exact_match**: a value of `1.0` indicates perfect agreement, any mismatch yields `0.0`.
