## Context

Python's `super()` follows the method resolution order (MRO) computed by C3
linearization. For a class $C$, the MRO is a tuple

$$
\mathrm{MRO}(C) = (C, B_1, B_2, \dots, \mathrm{object})
$$

that determines which implementation receives a cooperative `super()` call next.

A cooperative method can record its dispatch before forwarding:

```python
def method(self):
    log.append("super_dispatch:ClassName")
    super().method()
```

The sequence of these records follows the MRO order. Reconstructing the MRO from
the log requires preserving the observed dispatch sequence and adding the
implicit base class at the end when it is not logged.

## Task

Implement `reconstruct_mro(log)`.

The function receives a list of strings. Every dispatch record has the format
`"super_dispatch:ClassName"` and represents one class whose cooperative method
ran. Return a tuple containing the reconstructed MRO class names in order,
including `"object"` as the final entry.

Assume the log contains one dispatch entry for every class before `object`, with
no missing entries and no duplicate class names.

Example:

```python
log = [
    "super_dispatch:D",
    "super_dispatch:B",
    "super_dispatch:C",
    "super_dispatch:A",
]

result = reconstruct_mro(log)
# ("D", "B", "C", "A", "object")
```

## Example

```python
log = [
    "super_dispatch:Leaf",
    "super_dispatch:Mixin",
    "super_dispatch:Base",
]

print(reconstruct_mro(log))
# ("Leaf", "Mixin", "Base", "object")
```

## What the gate checks

The gate builds real Python class hierarchies and obtains their MROs from
CPython's `__mro__` attribute. It then creates dispatch logs from those actual
runtime MROs and checks that the submitted implementation reconstructs exactly
the same tuple of class names.

The `exact_match` score must be $1.0$ for all generated cases.
