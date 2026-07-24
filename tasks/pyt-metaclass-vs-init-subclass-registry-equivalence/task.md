## Context

Python class creation can be observed in multiple ways. A metaclass receives the class construction event before the class object is returned, while a base class can use `__init_subclass__` after a subclass has been created.

Both mechanisms can build the same registry. A registry maps each created subclass name to metadata about that class:

$$
R = \{ \text{name} \rightarrow (\text{name}, \text{bases}, \text{attributes}) \}.
$$

For a class $C$, the direct base classes are obtained from $C.__bases__$. Public attributes are the names defined in the class body that do not begin with an underscore.

## Task

Implement `build_registry()`:

```python
def build_registry() -> dict:
    ...
```

The function must create the same class hierarchy using two different registry mechanisms:

1. A metaclass that records subclasses when they are created.
2. A base class implementing `__init_subclass__` that records subclasses.

Return the registry produced by the `__init_subclass__` implementation.

The returned dictionary must contain entries for `Alpha` and `Beta`. Each entry must contain:

- `"name"`: the class name.
- `"bases"`: a list of direct base class names.
- `"attrs"`: sorted public attributes defined directly in the class.

The helper classes used to implement the registry must not appear in the returned result.

## Example

```python
result = build_registry()

result["Alpha"]
# {
#   "name": "Alpha",
#   "bases": ["Root"],
#   "attrs": ["kind"]
# }

result["Beta"]
# {
#   "name": "Beta",
#   "bases": ["Root"],
#   "attrs": ["size"]
# }
```

## What the gate checks

The gate creates an independent oracle implementation using real Python class creation behavior. It builds a registry once through a metaclass and once through `__init_subclass__`, then compares the candidate result with the registry from the oracle.

The `exact_match` metric must equal $1.0$.
