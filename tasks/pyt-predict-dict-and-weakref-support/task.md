## Context

Python instances can store attributes in an instance dictionary or use a more
compact layout through `__slots__`. A class without `__slots__` normally gives
instances a `__dict__`, while a slotted class may remove that storage.

Weak references are a separate capability. An object can be passed to
`weakref.ref` only when its class layout provides weak-reference support. A
slotted class usually needs a `"__weakref__"` slot to enable this feature.

The two properties form a layout classification:

$$
\text{layout}(obj) =
[\text{has\_dict}(obj),\text{supports\_weakref}(obj)] .
$$

These properties should be determined from actual object behavior rather than
only from a visual inspection of the class declaration.

## Task

Implement `predict_layouts()`:

```python
def predict_layouts():
    ...
```

Return a list of 20 two-element lists. Each inner list contains:

1. Whether an instance has an accessible `__dict__`.
2. Whether `weakref.ref(instance)` succeeds.

The returned classifications must match the classes used by the evaluator.

## Example

```python
result = predict_layouts()

# result contains 20 rows such as:
# [True, True]
# [False, False]

assert isinstance(result, list)
assert len(result) == 20
assert all(len(row) == 2 for row in result)
```

## What the gate checks

The gate creates 20 class layouts and computes the expected result using the
real CPython runtime:

- `vars(instance)` checks whether instance dictionary storage exists.
- `weakref.ref(instance)` checks weak-reference support.

The `exact_match` metric must equal $1.0$. The gate does not accept guessed
layout tables; the reference is computed from runtime behavior.
