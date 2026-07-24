## Context

Reverse-mode automatic differentiation records a computation as a Wengert list, also
called an autograd tape. Each node represents one operation and stores the nodes
that produced its inputs.

A computation graph is a directed acyclic graph (DAG). If a node $v$ depends on
nodes $u_1, u_2, \dots, u_k$, then every predecessor must appear earlier in the
linearized tape:

$$
u_1, u_2, \dots, u_k \prec v .
$$

A topological ordering converts the graph structure into an evaluation sequence.
Reverse-mode differentiation can later walk this sequence backwards to accumulate
gradients.

## Task

Implement `build_wengert_list(output_node)`.

The input is the final node of a scalar expression graph. Each node object has:

- `name`: a unique string identifier.
- `op`: the operation name, such as `"leaf"`, `"add"`, or `"mul"`.
- `inputs`: a list of predecessor nodes used to compute this node.

Return a list of dictionaries representing the Wengert list in topological order.
Only nodes reachable from `output_node` should be included.

Each dictionary must contain:

```python
{
    "name": node.name,
    "op": node.op,
    "inputs": [input_node.name, ...]
}
```

The ordering must be deterministic. When multiple nodes are ready, visit them
according to the order they are encountered in the graph traversal.

The function must not modify the input graph.

## Example

```python
class Node:
    def __init__(self, name, op, inputs=None):
        self.name = name
        self.op = op
        self.inputs = [] if inputs is None else inputs

x = Node("x", "leaf")
y = Node("y", "leaf")
z = Node("z", "add", [x, y])

tape = build_wengert_list(z)

# [
#   {"name": "x", "op": "leaf", "inputs": []},
#   {"name": "y", "op": "leaf", "inputs": []},
#   {"name": "z", "op": "add", "inputs": ["x", "y"]}
# ]
```

## What the gate checks

The gate builds several computation DAGs and independently computes a valid
canonical topological Wengert list from the graph structure. The returned list
must exactly match the oracle ordering, including node fields and ordering.

A traversal that places a node before its dependencies, skips shared nodes, or
uses non-deterministic ordering will fail.
