## Context

In CPython, every class object is represented by a type object. The type object
contains a `tp_flags` bit field exposed in Python as `T.__flags__`. One bit in
this field, `Py_TPFLAGS_HEAPTYPE`, indicates whether the type object was
allocated dynamically on the heap.

For a type object $T$, the heap-type predicate is

$$
\mathrm{is\_heap}(T) = (T.__flags__ \mathbin{\&} M) \neq 0,
$$

where $M$ is the heap-type flag mask.

The distinction is about how the type object itself was created. Most
user-defined Python classes are heap types, while many built-in CPython types
are static types.

The type relationship can be viewed as

$$
\mathrm{object} \rightarrow \mathrm{type(object)}
\rightarrow \mathrm{type}(\mathrm{type(object)}).
$$

This task receives type objects directly and classifies them from their CPython
metadata.

## Task

Implement `classify_heap_types(types)`:

```python
def classify_heap_types(types):
    ...
```

The function receives a list of Python type objects and returns a list of
booleans with the same length.

For each type object `T`, return `True` if its `__flags__` contains the
`Py_TPFLAGS_HEAPTYPE` bit and `False` otherwise. Use the flag information
directly. Do not classify by module names, class names, inheritance patterns,
or a fixed list of known classes.

## Example

```python
class UserClass:
    pass

result = classify_heap_types([
    int,
    dict,
    UserClass,
])

# [False, False, True]
```

## What the gate checks

The gate creates several static CPython types and dynamically created heap
types. It computes the expected booleans from each type object's `__flags__`
field and the heap-type mask, then compares the returned list exactly.

The test set includes a heap type whose module name is `"builtins"` so that
module-based heuristics fail. Only reading the actual type flags produces the
correct result.
