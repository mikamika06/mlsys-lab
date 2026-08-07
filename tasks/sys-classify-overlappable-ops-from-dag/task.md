## Context

A training step's operations form a dependency DAG: node $u \to v$ means
op $u$ must finish before op $v$ can start. Every op is either a
**compute** op (runs on the compute engine, e.g. a matmul) or a **comm**
op (runs on the network/copy engine, e.g. an all-reduce or all-gather).
Because compute and comm run on physically different engines, a
scheduler can overlap them **in time** — but only if doing so doesn't
violate a dependency: comm op $c$ and compute op $k$ can overlap iff
neither is a (transitive) prerequisite of the other, i.e. there is no
directed path from $c$ to $k$ and no directed path from $k$ to $c$ in the
DAG. If one is reachable from the other, they are ordered and cannot run
concurrently.

Formally, let $\mathrm{Desc}(u)$ be the set of nodes reachable from $u$
by following edges forward. Ops $u$ and $v$ are **overlappable** iff

$$
v \notin \mathrm{Desc}(u) \ \text{ and }\ u \notin \mathrm{Desc}(v).
$$

## Task

Implement `overlappable_ops(types, edges)`:

```python
def overlappable_ops(types: list[str], edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    ...
```

- `types`: list of `N` strings, each `"compute"` or `"comm"`, indexed
  `0..N-1` — the id of each op is its position in this list.
- `edges`: list of `(u, v)` pairs, each meaning op `u` must complete
  before op `v` starts. The graph is guaranteed to be a DAG (no cycles).

Return the set of every `(comm_id, compute_id)` pair (comm id first,
compute id second) such that the comm op and the compute op are
overlappable per the definition above.

## Example

```python
types = ["compute", "comm", "compute", "comm", "compute"]
edges = [(0, 1), (1, 4), (2, 3)]
overlappable_ops(types, edges)
# {(1, 2), (3, 0), (3, 4)}
#
# chain 0 -> 1 -> 4: comm 1 is ordered against compute 0 and compute 4
#   (both are on its dependency chain), so neither pair is in the result.
# chain 2 -> 3: comm 3 is ordered against compute 2, so (3, 2) is excluded.
# there is no path connecting {0, 1, 4} to {2, 3} at all, so comm 1
#   overlaps compute 2, and comm 3 overlaps both compute 0 and compute 4.
```

## What the gate checks

The grader runs the fixed example above plus several deterministically
generated random DAGs (`random.Random` seeded, `N` between 6 and
16 nodes, edges only from a lower index to a higher index so the graph is
always acyclic, roughly a third of nodes marked `"comm"`) through an
independent oracle that computes forward reachability from every node via
breadth-first search and checks the two-way non-reachability condition
above.

`exact_match` is `1.0` only if your returned set equals the oracle's set
exactly on **every** case, else `0.0`. Only checking one direction of
reachability (e.g. missing that `k` can be an ancestor of `c` even when
`c` is not an ancestor of `k`), including compute-compute or comm-comm
pairs in the output, or using direct-edge adjacency instead of full
transitive reachability will all produce a mismatched set.
