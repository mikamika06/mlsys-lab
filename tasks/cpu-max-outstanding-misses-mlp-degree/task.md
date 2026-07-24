## Context

Pointer chasing (`p = p->next`) is fully serial *within one chain*: you
cannot compute the address of the next load until the previous one's
result comes back, so a single chain never benefits from a CPU's ability
to have several cache misses outstanding at once. But real code rarely
walks just one chain — running $k$ *independent* chains side by side (or
processing several unrelated tree branches, hash buckets, or linked lists
in the same loop) lets up to $k$ misses be in flight simultaneously, even
though each individual chain is still serialized internally. That
achievable overlap is the load's **memory-level parallelism (MLP)**.

Model this as a DAG: one node per load, one directed edge
$u \to v$ whenever load $v$'s address depends on load $u$'s result (so $v$
cannot issue until $u$ completes). Under **ASAP scheduling** with
unlimited outstanding-miss buffers — issue a load the instant every
dependency it has has completed — every node has a well-defined
dependency depth $\text{depth}(v)$: $0$ if $v$ has no incoming edges,
otherwise $1 + \max(\text{depth}(u))$ over every edge $u \to v$. Since
every load takes the same fixed latency, all nodes at the same depth are
in flight during the exact same time window. The maximum achievable MLP
is therefore the width of the DAG's widest depth level:

$$\text{mlp\_degree} = \max_{d \ge 0} \; \bigl|\{v : \text{depth}(v) = d\}\bigr|$$

## Task

Implement:

```cpp
int mlp_degree(int n, const int* dep_from, const int* dep_to, int num_edges);
```

`n` loads, numbered `0..n-1`. `num_edges` dependency edges: edge `i` means
load `dep_to[i]` cannot issue until load `dep_from[i]` has completed. The
edges form a DAG (no cycles, but not necessarily given in any particular
order). Compute each load's dependency depth as above, and return the size
of the largest same-depth group.

## Example

The driver (`main.cpp`, fixed) builds 4 independent pointer-chase chains
of lengths 5, 5, 5, and 3 — internally fully sequential, but with no
dependencies *between* chains:

```
chain A: 0 -> 1 -> 2 -> 3 -> 4     (depths 0,1,2,3,4)
chain B: 5 -> 6 -> 7 -> 8 -> 9     (depths 0,1,2,3,4)
chain C: 10 -> 11 -> 12 -> 13 -> 14 (depths 0,1,2,3,4)
chain D: 15 -> 16 -> 17            (depths 0,1,2)
```

At depth 0, all 4 chains have a ready node (width 4); same at depths 1 and
2. Chain D has ended by depth 3, leaving only 3 chains active at depths 3
and 4 (width 3). The widest level is 4:

```
mlp_degree=4
```

4 independent streams cap the achievable overlap at 4-way MLP, regardless
of how long any individual chain runs.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires the printed `mlp_degree` to `exact_match` the same
driver linked against the reference derivation. Counting only the total
number of loads, or the length of the single longest chain (5, from chains
A/B/C), instead of the widest same-depth group, misses the actual
parallelism the hardware could exploit and fails the gate. The starter
returns `0`.
