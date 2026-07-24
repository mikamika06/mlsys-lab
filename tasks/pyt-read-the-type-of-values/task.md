## Context

Every Python value has an associated type object. The built-in function `type(x)` returns the type object of `x`, and the type object's `__name__` attribute gives its readable class name.

For a value $x$, the operation performed by Python is:

$$
\mathrm{type\_name}(x) = \mathrm{type}(x).__name__ .
$$

Python objects also form a type/instance/metatype triangle: ordinary values are instances of types, and type objects themselves are instances of the metaclass `type`.

## Task

Implement `read_type_names(values)`:

```python
def read_type_names(values):
    ...
```

The function receives an iterable of Python values and returns a list containing the type name of each value in the same order.

Use the actual runtime type information from Python. Do not convert values to strings or infer types from their contents.

## Example

```python
values = [1, 2.5, "hello", [1, 2]]
names = read_type_names(values)

# ["int", "float", "str", "list"]
```

## What the gate checks

The gate builds a set of Python values containing different built-in types and compares the returned list against a reference computed by the CPython runtime using `type(value).__name__` for every value.

The result must exactly match the runtime type names.
