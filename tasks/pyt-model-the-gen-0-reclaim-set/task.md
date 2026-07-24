## Context

Reference counting reclaims an object immediately when its reference count reaches
zero. A cycle cannot be reclaimed this way because every object in the cycle still
has an incoming reference from another object in the same cycle.

A generational cycle collector finds unreachable cycles by considering the
references between tracked objects. For a set of candidate objects, define the
internal incoming reference count as

$$
I(v) = |\{u \mid u \rightarrow v,\ u \text{ is in the candidate set}\}|.
$$

The collector computes a trial count

$$
T(v) = \mathrm{refcount}(v) - I(v).
$$

Objects with $T(v) > 0$ are reachable from outside the candidate subgraph.
After removing those objects, the remaining objects are unreachable cycles and
can be reclaimed.

## Task

Implement `reclaim_set(nodes, edges, refcounts, roots)`.

Arguments:

- `nodes` is a list of integer object ids tracked by gen-0.
- `edges` is a dictionary mapping each node id to a list of node ids it references.
- `refcounts` is a dictionary mapping node ids to their total reference count.
- `roots` is a list of node ids that are externally reachable.

Return a `set` containing the ids of nodes that a single gen-0 cycle collection
would reclaim.

The implementation should model the cycle collection step only. It does not need
to mutate the graph or return collection order.

## Example

```python
nodes = [1, 2, 3, 4]
edges = {1: [2], 2: [1], 3: [4], 4: []}
refcounts = {1: 1, 2: 1, 3: 1, 4: 0}
roots = [3]

reclaim_set(nodes, edges, refcounts, roots)
# {1, 2}
```

## What the gate checks

The gate runs the implementation on several heap graphs. It independently
computes the trial-deletion reference algorithm and requires the returned set of
ids to exactly match the computed reclaim set.

A correct implementation must distinguish external reachability from references
that only exist inside an unreachable cycle.
