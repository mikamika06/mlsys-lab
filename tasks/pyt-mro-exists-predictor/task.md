## Context

Python uses the C3 linearization algorithm to determine a class method resolution order (MRO) when multiple inheritance is used. A class with direct bases $B_1, B_2, \dots, B_k$ has an MRO computed from the merge operation:

$$
MRO(C) = [C] + merge(MRO(B_1), MRO(B_2), \dots, MRO(B_k), [B_1, B_2, \dots, B_k]).
$$

The merge step repeatedly selects a candidate head that does not appear in the tail of any other sequence. If no valid candidate exists, the hierarchy is inconsistent and CPython rejects the class definition with a `TypeError`.

This task asks you to predict whether CPython can construct the requested class hierarchy. The input is an encoded graph where each integer represents a class node and each list contains direct bases in declaration order.

## Task

Implement `mro_exists(graph)`:

```python
def mro_exists(graph: list[list[int]]) -> bool:
    ...
```

The function receives a list of direct-base lists. Node `0` is the class being tested. For every node `i`, `graph[i]` contains the bases of class `i`.

Return `True` when CPython can create node `0` with these bases, and return `False` when C3 linearization would fail.

Do not create real classes with `type()`. Compute the result by simulating the C3 feasibility rules.

## Example

```python
graph = [
    [1, 2],
    [],
    [],
]

assert mro_exists(graph) is True
```

A conflict can occur when parent order requirements cannot be merged:

```python
graph = [
    [1, 2, 3],
    [3, 2],
    [],
    [],
]

assert mro_exists(graph) is False
```

## What the gate checks

The gate builds several encoded class graphs and uses real CPython class creation as the oracle. It constructs equivalent classes with `type(name, bases, {})`.

If CPython creates the class, the oracle result is `True`. If CPython raises `TypeError`, the oracle result is `False`.

The gate compares `mro_exists` against this oracle with exact boolean matching. A correct implementation must reproduce C3 existence checks.
