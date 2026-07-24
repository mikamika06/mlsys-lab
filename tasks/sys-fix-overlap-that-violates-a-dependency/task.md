## Context

A runtime scheduler may overlap computation and communication to reduce idle time. An overlap is only valid when every operation appears after the operations it depends on.

Represent a dependency graph as a set of operations. Each operation has a unique name and a list of names that must complete before it can start. A schedule is a permutation of all operation names.

For a schedule $S = (s_1, s_2, \dots, s_n)$, an operation $s_i$ is valid only if every dependency of $s_i$ appears in an earlier position. If $p(x)$ is the position of operation $x$, the dependency rule is:

$$
p(d) < p(x) \quad \text{for every dependency } d \text{ of } x .
$$

A broken overlap optimization can place communication before its producer, causing a dependency violation. The fix is to produce a legal schedule while preserving the original order whenever possible.

## Task

Implement `fix_schedule(ops)`:

```python
def fix_schedule(ops):
    ...
```

`ops` is a list of tuples:

```python
(name, dependencies)
```

where `name` is a string and `dependencies` is a list of operation names.

Return a list of operation names representing a valid execution order.

The returned schedule must be a stable topological ordering: when multiple operations are currently ready, choose the one that appeared earlier in the input list. All input operations must appear exactly once in the result.

## Example

```python
ops = [
    ("comm_grad", ["compute_grad"]),
    ("compute_grad", []),
    ("update", ["comm_grad"]),
]

fix_schedule(ops)
# ["compute_grad", "comm_grad", "update"]
```

`comm_grad` originally overlapped before its producer, but the fixed schedule waits for `compute_grad`.

## What the gate checks

The gate computes a reference stable topological scheduler from the dependency graph and compares the submitted result exactly.

The result must match the oracle order, contain every operation once, and satisfy every dependency constraint. A scheduler that only swaps one visible violation or uses a different tie-breaking rule will fail.
