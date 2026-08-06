## Context

Python class membership checks can be understood as a graph traversal over method resolution order information. A type has an MRO containing the classes searched when resolving attributes. If an object's class is connected to a target class through this ordering, `isinstance` reports membership.

In this task, the MRO information is represented as an adjacency matrix $M \in \{0,1\}^{n \times n}$. If $M_{ij}=1$, class $j$ appears directly in the MRO walk reachable from class $i$. The diagonal represents that every class is reachable from itself.

For a starting class index $c$, the reachable set is the transitive closure of the directed graph:

$$
R(c) = \{x \mid \text{there is a path from } c \text{ to } x\}.
$$

A pair $(o, t)$ should be classified as true when the target class $t$ is in the reachable set of the object's class $o$.

## Task

Implement `mro_isinstance(mro_adj, pairs)`:

```python
def mro_isinstance(mro_adj: list[list[int]], pairs: list[list[int]]) -> list[bool]:
    ...
```

The inputs are:

- `mro_adj`: a square integer list describing class MRO edges.
- `pairs`: a two-column integer list. Each row contains `(obj_class_index, target_class_index)`.

Return a boolean list with one element per row of `pairs`. The result at position $i$ must indicate whether the target class is reachable from the object class in the supplied MRO graph.

Only the supplied graph may be used. Do not call Python `isinstance`, inspect real classes, or use object metadata.

## Example

```python

mro = [
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
]

pairs = [
    [0, 1],
    [3, 0],
    [2, 1],
]

print(mro_isinstance(mro, pairs))
# [ True False  True]
```

## What the gate checks

The gate computes the expected membership vector by performing graph reachability over the supplied adjacency matrix and compares it with the submitted function output.

The returned array must exactly match the oracle result. The gate checks multiple graphs containing inheritance chains, branches, self edges, and disconnected classes.
