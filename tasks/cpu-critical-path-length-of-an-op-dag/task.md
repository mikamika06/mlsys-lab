## Context

A compiler or CPU scheduler can execute independent operations in parallel, but a
dependency chain limits how quickly a group of operations can complete. The
critical path of an operation DAG is the longest dependency path weighted by
operation latency.

For a DAG with operations $V$ and directed dependency edges $E$, each operation
$v \in V$ has latency $l(v)$. The earliest completion time of an operation is

$$
C(v) = l(v) + \max_{u \rightarrow v} C(u),
$$

where the maximum over an empty set is $0$. The critical-path length is

$$
\max_{v \in V} C(v).
$$

This value represents the minimum number of serialized latency units required if
all independent operations are scheduled with unlimited parallel execution.

## Task

Implement `critical_path_length(nodes, edges)`:

```python
def critical_path_length(nodes, edges):
    ...
```

`nodes` is a dictionary mapping integer operation IDs to positive integer
latencies. `edges` is a list of `(src, dst)` pairs where `src` must complete
before `dst` can start. The input is a valid directed acyclic graph.

Return the integer critical-path length of the DAG.

## Example

```python
nodes = {
    0: 3,
    1: 5,
    2: 2,
    3: 4,
}

edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
]

critical_path_length(nodes, edges)
# 12
```

The longest path is $0 \rightarrow 1 \rightarrow 3$ with latency
$3 + 5 + 4 = 12$.

## What the gate checks

The gate compares the returned critical-path length against an independently
computed dynamic-programming reference on several DAGs. The result must be an
exact integer match.
