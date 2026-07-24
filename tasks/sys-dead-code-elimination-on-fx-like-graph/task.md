## Context

In many deep learning frameworks a computation is represented as a directed acyclic graph (DAG) of operations. A node may depend on the outputs of other nodes, and only those nodes that contribute to the final outputs are needed for execution or compilation. Removing nodes that cannot influence any output is called *dead‑code elimination*. It reduces memory usage and improves performance.

We model a tiny FX‑like graph as a list of node dictionaries:

```python
{
    "id": int,          # unique identifier
    "op": str,          # operation name (e.g. "add", "mul")
    "inputs": List[int] # ids of predecessor nodes
}
```

The `outputs` argument is a list of node ids that are considered final outputs of the graph.

## Task

Implement `dead_code_elimination(nodes, outputs)`:

```python
def dead_code_elimination(
    nodes: List[Dict[str, Any]],
    outputs: List[int]
) -> Tuple[List[Dict[str, Any]], List[int]]:
    ...
```

The function must return a tuple `(new_nodes, new_outputs)` where

* `new_nodes` is the list of node dictionaries that are reachable from at least one id in `outputs`.  
  The nodes should be sorted by ascending `id`.
* `new_outputs` is the subset of the original `outputs` that remain present after pruning, also sorted.

The implementation must use only Python built‑in data structures and control flow; no external libraries.

## Example

```python
nodes = [
    {"id": 0, "op": "const",   "inputs": []},
    {"id": 1, "op": "add",     "inputs": [0]},
    {"id": 2, "op": "mul",     "inputs": [1]},
    {"id": 3, "op": "sub",     "inputs": [2]},   # reachable
    {"id": 4, "op": "noop",    "inputs": []},   # dead code
]
outputs = [3, 4]

new_nodes, new_outputs = dead_code_elimination(nodes, outputs)
print(new_nodes)
# [{'id': 0, 'op': 'const', 'inputs': []},
#  {'id': 1, 'op': 'add', 'inputs': [0]},
#  {'id': 2, 'op': 'mul', 'inputs': [1]},
#  {'id': 3, 'op': 'sub', 'inputs': [2]}]

print(new_outputs)
# [3]
```

Node `4` is removed because it cannot reach any output.

## What the gate checks

The grader computes a reference solution using the same graph representation and compares your result with it. The metric `exact_match` must be equal to `1.0`. No other metrics are evaluated.
