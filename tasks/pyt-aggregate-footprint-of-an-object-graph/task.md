## Context

Every Python object has memory overhead beyond the bytes used for its logical value. CPython objects include a universal header used by the runtime, and containers also store references to other objects.

A graph of Python objects therefore has a deep footprint that is not equal to the sum of its visible values. If multiple references point to the same object, the object should only be counted once. The deep footprint of a reachable object graph can be written as

$$
S(G) = \sum_{o \in R(G)} \operatorname{sizeof}(o),
$$

where $R(G)$ is the set of objects reachable from the graph roots and `sizeof` is the CPython `sys.getsizeof` result.

A flat numerical representation stores only the payload values in a contiguous list. Its footprint is

$$
S_{\mathrm{flat}} = \operatorname{nbytes}(\mathrm{list}).
$$

The ratio

$$
\rho = \frac{S(G)}{S_{\mathrm{flat}}}
$$

shows how much larger the Python object representation is compared with a compact numeric encoding.

## Task

Implement `aggregate_footprint(adjacency, payloads)`.

`adjacency` is a square Python integer matrix of shape $(n,n)$. Entry `adjacency[i, j]` is non-zero when object node $i$ references object node $j$.

`payloads` is a list of integer lists. Node $i$ stores its corresponding payload list.

The function must:

1. Build an object graph where every node is a Python dictionary containing its payload and child references.
2. Traverse all nodes reachable from the graph roots.
3. Compute the deep footprint by summing `sys.getsizeof` once per unique object identity.
4. Build a flat Python `int64` array containing all payload integers and compare against its `nbytes`.
5. Return the footprint ratio $\rho$ as a `float`.

The traversal must deduplicate objects using identity, not equality. Use only the Python standard library and Python.

## Example

```python

adjacency = [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
]

payloads = [[1, 2], [3], [4, 5]]

ratio = aggregate_footprint(adjacency, payloads)
# ratio is a float larger than 1.0
```

## What the gate checks

The gate creates several object graphs and computes the expected ratio using the real CPython object model with `sys.getsizeof`.

Your returned value is compared with that oracle. The `size_ratio` score is

$$
1 - \frac{|\rho_{\mathrm{candidate}}-\rho_{\mathrm{oracle}}|}{|\rho_{\mathrm{oracle}}|+10^{-12}},
$$

so only an implementation that performs the same identity-deduplicated deep size calculation passes.
