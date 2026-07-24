## Context

Every Python instance has memory overhead from its object layout. A normal class that
stores attributes in an instance dictionary keeps a separate mapping object, while a
class using `__slots__` stores declared attributes directly in slot storage.

For a fixed set of attributes, define the footprint ratio

$$
R = \frac{S_{\mathrm{dict}}}{S_{\mathrm{slots}}},
$$

where $S_{\mathrm{dict}}$ is the measured size of one dictionary-backed instance
including its `__dict__`, and $S_{\mathrm{slots}}$ is the measured size of one
slotted instance.

Python exposes the real object sizes through `sys.getsizeof`, which measures the
memory footprint reported by the CPython runtime for each object.

## Task

Implement `instance_footprint_ratio(attrs)`:

```python
def instance_footprint_ratio(attrs: dict) -> float:
    ...
```

The function receives a dictionary of attribute names and values. It must create two
equivalent classes:

- a normal class where the attributes are assigned on the instance
- a class using `__slots__` containing the provided attribute names

Create one instance of each class, assign all attributes, and return

$$
R = \frac{\mathrm{getsizeof}(\mathrm{dict\ instance}) +
\mathrm{getsizeof}(\mathrm{dict\ instance.__dict__})}
{\mathrm{getsizeof}(\mathrm{slots\ instance})}.
$$

The result must be a Python `float`.

## Example

```python
ratio = instance_footprint_ratio(
    {"x": 1, "y": 2, "name": "sample"}
)

# ratio is a positive float, usually greater than 1.0 on CPython
```

The exact value depends on the running CPython build, so do not hardcode a
constant.

## What the gate checks

The gate builds the reference ratio using CPython's own `sys.getsizeof` measurements
on the same attribute fixture. Your implementation passes when its returned value
matches the oracle result within the grader tolerance.

A solution that returns a fixed ratio or only measures one object instead of the
instance and its `__dict__` will fail.
