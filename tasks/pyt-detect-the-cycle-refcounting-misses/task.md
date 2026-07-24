## Context

CPython primarily uses reference counting to manage object lifetimes. A reference
count reaches zero when no object points to an object anymore, so an isolated
object can be freed immediately.

Reference counting alone cannot reclaim a reference cycle. For example, if
objects $a$ and $b$ point to each other,

$$
a \rightarrow b \rightarrow a,
$$

the objects keep each other's reference count above zero even when no outside
object points to the cycle.

A heap graph can be modeled as a directed graph $G=(V,E)$ where each node is an
object and each edge represents a strong reference. A strongly connected
component (SCC) is a maximal set of nodes where every node can reach every other
node. A component is a reference-counting leak candidate when:

1. it contains a cycle, meaning its SCC size is greater than one or it has a
   self-loop;
2. it has no incoming edge from outside the component.

Such a component is isolated from the rest of the heap but cannot be destroyed
by simple reference counting.

## Task

Implement `uncollectable_cycles(graph)`:

```python
def uncollectable_cycles(graph: dict[int, list[int]]) -> list[list[int]]:
    ...
```

The input is a directed heap graph. Keys are node identifiers and values are
lists of nodes referenced by that node. Every referenced node appears as a key.

Return all SCCs that represent cycles with no external incoming edges.

The returned value must:

- contain one sorted list of node ids per qualifying SCC;
- sort nodes inside each component increasingly;
- sort the list of components by their first node id increasingly;
- exclude acyclic isolated nodes.

Do not mutate the input graph.

## Example

```python
graph = {
    1: [2],
    2: [1],
    3: [4],
    4: [],
    5: [5],
    6: [5],
}

uncollectable_cycles(graph)
# [[1, 2]]
```

The nodes `1` and `2` form an isolated cycle. Node `5` has a self-cycle, but
node `6` points into it, so it has an external incoming edge and is not returned.

## What the gate checks

The gate computes the expected result using an independent graph traversal
implementation and compares the returned SCC node sets exactly. The tests cover
mutual references, self-cycles, incoming references, disconnected heaps, and
larger strongly connected components.

The `exact_match` score must equal $1.0$.
