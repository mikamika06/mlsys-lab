## Context

During class creation, Python calls the metaclass method `__prepare__` before executing the class body. The returned mapping receives assignments as the class body runs.

A normal class dictionary only shows the final state of each key. If a class body assigns the same name more than once, the final namespace cannot reveal the full assignment sequence.

A recording mapping can observe every assignment event. For class body assignments $x_1, x_2, \dots, x_n$, the captured order is

$$[x_1, x_2, \dots, x_n].$$

The capture must ignore interpreter-provided entries such as `__module__` and `__qualname__`.

## Task

Implement `capture_class_body_order(names)`.

The function receives a list of valid Python identifier strings. Create a temporary class using a metaclass with `__prepare__`. The prepared namespace must record every assignment made by the class body, including repeated assignments to the same name.

The function should return the list of assignment keys observed by the prepared mapping.

```python
def capture_class_body_order(names: list[str]) -> list[str]:
    ...
```

The generated class body should assign each name in `names` in order. Repeated names are meaningful because they must appear repeatedly in the returned list.

## Example

```python
capture_class_body_order(["alpha", "beta", "alpha"])
```

returns:

```python
["alpha", "beta", "alpha"]
```

## What the gate checks

The gate builds the expected result using a real CPython class creation path with a custom `__prepare__` mapping that records `__setitem__` calls.

The `exact_match` metric must equal $1.0$. Implementations that inspect the final class dictionary fail because repeated assignments have already been collapsed.
