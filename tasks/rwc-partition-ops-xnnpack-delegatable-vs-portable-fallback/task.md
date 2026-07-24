## Context

In many deep‑learning runtimes, a graph of operations is split into two parts:
operations that can be executed by an external delegate (e.g., XNNPACK) and those
that must fall back to the portable implementation.  
Let  

$$S \subseteq \mathcal{O}$$

be the set of operation names that the delegate supports, where $\mathcal{O}$ is the
universe of all possible op names.  For each node $n$ in a computation graph we
must decide whether it can be delegated:

$$\text{delegate}(n) = \begin{cases}
\texttt{True} & \text{if } \operatorname{name}(n)\in S,\\[4pt]
\texttt{False}& \text{otherwise.}
\end{cases}$$

The task is to implement this decision logic for a list of graph nodes.

## Task

Implement the function `partition_ops`:

```python
from typing import List, Tuple, Set, Dict

def partition_ops(
    nodes: List[Tuple[int, str]],
    delegate_support: Set[str]
) -> Dict[int, bool]:
    ...
```

* `nodes` is a list of `(node_id, op_name)` tuples.  
  Node IDs are unique integers.
* `delegate_support` is the set $S$ of operation names that can be delegated.
* The function must return a dictionary mapping each node ID to a boolean
  indicating whether the node should be delegated (`True`) or fall back
  (`False`).  

The implementation must be pure and efficient; it will be graded against an
oracle that applies the same rule.

## Example

```python
nodes = [(0, 'conv2d'), (1, 'relu'), (2, 'add')]
delegate_support = {'conv2d', 'add'}

result = partition_ops(nodes, delegate_support)
# result == {0: True, 1: False, 2: True}
```

## What the gate checks

The grader computes a reference mapping using the same rule and compares it
exactly to your output.  The metric is `exact_match`; any mismatch yields a
score of `0.0`.  No other constraints are imposed.
