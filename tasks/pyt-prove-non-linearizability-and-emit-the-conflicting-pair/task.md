## Context

Python uses the C3 algorithm to compute a class method resolution order (MRO). A class can only be created when all precedence constraints can be merged into one consistent linear order.

For a class with base lists $L_1, L_2, \dots, L_k$, C3 repeatedly selects a valid head element from the merge:

$$
\operatorname{MRO}(C) = [C] + \operatorname{merge}(L_1, L_2, \dots, L_k, [B_1, B_2, \dots, B_k]).
$$

A candidate head $h$ is valid only if it does not appear in the tail of any remaining list. If every available head appears in another tail, the constraints are contradictory and the hierarchy is not linearizable.

CPython reports this situation when creating the class. The error identifies the pair of bases whose relative ordering cannot be satisfied.

## Task

Implement `conflicting_pair(spec)`:

```python
def conflicting_pair(spec):
    ...
```

`spec` describes a hierarchy as a list of class declarations. Each declaration is a tuple:

```python
(class_name, (base_name_1, base_name_2, ...))
```

Base names refer to classes declared earlier in `spec`. The hierarchy is guaranteed to be non-linearizable.

Return a tuple containing the two base names reported by CPython as the contradictory pair.

The returned tuple must preserve the order used by CPython's error message.

## Example

```python
spec = [
    ("A", ()),
    ("B", ()),
    ("X", ("A", "B")),
    ("Y", ("B", "A")),
    ("Z", ("X", "Y")),
]

conflicting_pair(spec)
# ("A", "B")
```

## What the gate checks

The gate builds real Python classes with `type()` and uses CPython's own MRO construction failure as the oracle. The returned pair is compared exactly with the pair extracted from the CPython `TypeError` message.

A correct solution must identify the actual conflicting constraint, not just return the first two bases or the final class's direct parents.
