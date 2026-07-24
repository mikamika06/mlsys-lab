## Context

Python method resolution order (MRO) uses the C3 linearization algorithm to combine a class with the linearizations of its base classes while preserving local precedence ordering and monotonicity.

For a class $C$ with direct bases $B_1, B_2, \dots, B_k$, C3 computes:

$$
L(C) = [C] + merge(L(B_1), L(B_2), \dots, L(B_k), [B_1, B_2, \dots, B_k]).
$$

The `merge` operation repeatedly chooses a valid head. A candidate is valid when it appears at the front of one sequence and does not appear anywhere in the tails of the other sequences. If no candidate satisfies this rule, the hierarchy is inconsistent.

This task represents a class hierarchy as an adjacency matrix. Entry $A_{ij}=1$ means class $i$ directly inherits from class $j$. The class names are provided separately.

## Task

Implement `c3(graph, cls_index, names)`.

Arguments:

- `graph`: a NumPy integer matrix where `graph[i, j] == 1` means class `i` has direct base class `j`.
- `cls_index`: the index of the class whose linearization should be returned.
- `names`: a list of class names indexed by the rows of `graph`.

Return a list of names representing the C3 linearization of the selected class. The result must match Python's `__mro__` order for valid hierarchies.

Do not use `type.__mro__`, `inspect`, or other built-in MRO helpers. Implement the merge procedure yourself.

## Example

```python
import numpy as np

graph = np.array([
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
])

names = ["D", "B", "C", "A"]

c3(graph, 0, names)
# ["D", "B", "C", "A"]
```

## What the gate checks

The gate builds real Python class hierarchies and uses Python's own `__mro__` attribute as the oracle. It converts those hierarchies into adjacency matrices and checks that `c3` returns the same name sequence.

The implementation is tested on multiple inheritance patterns, including shared ancestors and deeper hierarchies. Any mismatch fails the exact match gate.
