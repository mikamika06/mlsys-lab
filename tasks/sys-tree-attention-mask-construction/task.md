## Context

Speculative decoding methods can evaluate multiple candidate continuations arranged as a tree. Each node in the tree represents a token position, and a node should only attend to tokens on its path back to the root.

For a tree with $n$ nodes, define a parent array $p$ where $p_i$ is the parent of node $i$. The root has parent $-1$. The attention mask $M \in \{0,1\}^{n \times n}$ uses the convention

$$
M_{ij} =
\begin{cases}
1 & \text{if node } i \text{ can attend to node } j,\\
0 & \text{otherwise.}
\end{cases}
$$

A node can attend to itself and every ancestor. If the ancestor chain of node $i$ is

$$
i, p_i, p_{p_i}, \dots, 0,
$$

then all of those positions receive value $1$ in row $i$ of the mask. This is a tree-shaped version of causal attention.

## Task

Implement `build_tree_attention_mask(parents)`:

```python
def build_tree_attention_mask(parents: list[int]) -> np.ndarray:
    ...
```

The input is a list of length $n$. `parents[i]` gives the parent index of node `i`, or `-1` for the root. Return an $n \times n$ NumPy array with dtype `int64` where row `i` marks all ancestors of node `i` (including itself) with `1`.

Assume the input describes a valid tree: exactly one root exists and every non-root parent index is smaller than its child index.

## Example

```python
parents = [-1, 0, 0, 2, 3]

mask = build_tree_attention_mask(parents)

# [[1 0 0 0 0]
#  [1 1 0 0 0]
#  [1 0 1 0 0]
#  [1 0 1 1 0]
#  [1 0 1 1 1]]
```

## What the gate checks

The gate builds the expected mask using an independent ancestor-traversal oracle and compares the returned array exactly. The result must match the oracle values, shape, and integer mask representation.
