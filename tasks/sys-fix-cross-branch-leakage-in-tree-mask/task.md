## Context

Tree-based speculative decoding methods generate multiple candidate continuations from a shared prefix. The attention mask must allow a token to attend to tokens on the same verified path and to shared ancestors, but it must prevent attention to tokens that belong only to sibling branches.

Represent the generated tree with a parent array. Node $0$ is the root and every other node $i$ has exactly one parent $p_i$ with $0 \le p_i < i$. The ancestor relation defines the visible history of each node.

For nodes $i$ and $j$, the correct tree attention rule is:

$$
M_{ij} =
\begin{cases}
1 & \text{if } j \text{ is an ancestor of } i \text{ or } j=i,\\
0 & \text{otherwise.}
\end{cases}
$$

A leaked mask incorrectly sets some entries to $1$ when $j$ is a sibling-branch token. This allows a speculative token to consume information from an unrelated candidate path.

## Task

Implement `build_tree_mask(parents)`:

```python
def build_tree_mask(parents):
    ...
```

The input is a Python list of integers. `parents[0]` is always `-1` for the root. For every other index `i`, `parents[i]` is the parent index of node `i`.

Return a NumPy array of shape `(n, n)` with dtype `int8`. Entry `(i, j)` must be `1` exactly when node `j` is visible to node `i` under the ancestor rule above.

Do not use any external ML libraries.

## Example

```python
parents = [-1, 0, 0, 1, 1, 2]

mask = build_tree_mask(parents)

# Node 4 can see: root -> 1 -> 4
# Row 4 is:
# [1, 1, 0, 0, 1, 0]
```

## What the gate checks

The gate builds several trees and computes the expected mask with an independent ancestor-walk oracle. The returned mask must exactly match the oracle output.

The check catches cross-branch leakage by including trees where sibling nodes share the same parent. A solution that only allows the root or all previous nodes to attend will not pass.
