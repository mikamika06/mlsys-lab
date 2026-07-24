## Context

A **traced graph** is a static representation of a computation recorded by intercepting every operation during eager execution. Tools like `torch.fx` or `jax.make_jaxpr` capture a Directed Acyclic Graph (DAG) whose nodes are operations and edges are data dependencies. Once captured, the graph can be transformed for performance.

**Constant folding** is a classic compiler optimisation: if all the inputs of an operation are compile‑time constants, the operation can be replaced by its pre‑computed result.  The graph becomes smaller and run‑time redundant work is eliminated.

In this task you are given a minimal graph IR that supports the integer operations `add`, `mul`, `sub`, `neg`, together with constant literals and named *input* leaves.  You must implement constant folding: walk the graph in topological order, and for every node whose *entire* sub‑graph of dependencies is constant, replace that node with a constant node holding the computed value.

## Task

Implement `constant_fold(graph: dict) -> dict`.  

The **input** `graph` is a dict with two keys:

* `"nodes"`:  a `dict[int, dict]` mapping an integer id to a node dictionary.  
  Each node dictionary has at least:
  * `"op"`: a string in `{"add", "mul", "sub", "neg", "constant", "input"}`.
  * `"inputs"`: a `list[int]` of node ids that this node reads from.  For `"constant"` and `"input"` this list is empty.
  * If `"op" == "constant"` the node also contains `"value": int`.
  * If `"op" == "input"` the node also contains `"name": str` (the variable name).  
* `"output"`: an `int` – the id of the node whose value is the graph’s output.

The nodes are guaranteed to be topologically sorted: every parent id is strictly smaller than its child id, so iterating ids in increasing order gives a valid topological order.

**Constant‑folding rules** – for each node in increasing id order:

1. If `"op"` is `"constant"` or `"input"` – do nothing.
2. Otherwise, check whether **all** input nodes already carry a `"value"` key.  
   (A node obtains a `"value"` key if it is a constant literal or has been folded earlier.)
3. If yes, compute the result of the operation using the integer values of its inputs:
   * `"add"`  → addition    (two inputs)
   * `"mul"`  → multiplication (two inputs)
   * `"sub"`  → subtraction (first minus second)
   * `"neg"`  → negation    (one input)
4. Then replace the node:
   * set `"op"` to `"constant"`
   * set `"value"` to the computed integer
   * set `"inputs"` to `[]` (the old input ids are no longer needed for this node).
5. Leave nodes that cannot be folded completely unchanged (their `"op"`, `"inputs"`, and any other keys remain as they were).

The **return** value must be a dict in the same format (with the same `"output"`).  
You are allowed to mutate the input, but the simplest approach is to first make a deep copy using `copy.deepcopy`.

## Example

```python
graph = {
    "nodes": {
        0: {"op": "constant", "inputs": [], "value": 2},
        1: {"op": "constant", "inputs": [], "value": 3},
        2: {"op": "add",      "inputs": [0, 1]},
        3: {"op": "sub",      "inputs": [2, 1]},
    },
    "output": 3,
}

folded = constant_fold(graph)

# Expected result:
# {
#     "nodes": {
#         0: {"op": "constant", "inputs": [], "value": 2},
#         1: {"op": "constant", "inputs": [], "value": 3},
#         2: {"op": "constant", "inputs": [], "value": 5},
#         3: {"op": "constant", "inputs": [], "value": 2},
#     },
#     "output": 3,
# }
```

A mixed example with an input variable:

```python
graph = {
    "nodes": {
        0: {"op": "input", "inputs": [], "name": "x"},
        1: {"op": "constant", "inputs": [], "value": 10},
        2: {"op": "mul", "inputs": [0, 1]},   # 10 * x  – not foldable
    },
    "output": 2,
}
# The result is identical to the input (node 2 stays a mul).
```

## What the gate checks

**One gate – `exact_match`**.  
Your output graph (the dict returned by `constant_fold`) must be exactly equal to the reference output for every hidden test graph.  “Exactly equal” means the whole Python structure (`dict` of `dict`s, `list`s, `int`s, `str`s) compares equal with `==`.  Any deviation – a wrong `"value"`, a missing `"op"` change, leftover inputs on a folded node, or an unintended mutation of the original graph – causes the gate to return `0.0`.  The only way to pass is to follow the constant‑folding procedure above verbatim.
