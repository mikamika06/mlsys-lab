## Context

Speculative decoding methods can arrange proposed tokens in a tree. Each node represents a possible next token and has an independent probability that the proposed token is accepted.

For a node with acceptance probability $p$, the expected number of accepted tokens contributed by that node depends on whether all previous nodes on the path were accepted. A child can only contribute if every ancestor was accepted.

Consider a rooted tree where the root has depth $0$. For each non-root node $v$, let $p_v$ be its acceptance probability. The accepted length counts accepted nodes along a single sampled path. The expected accepted length is:

$$
E = \sum_{v \in \text{nodes except root}} \prod_{u \in \text{path(root, v)}} p_u .
$$

This means each node contributes its probability of being reached and accepted. A wide tree can increase expected accepted length because multiple branches represent alternative continuations.

The tree is represented as an adjacency list. The root is node `0`, and node indices are integers from `0` to `n-1`.

## Task

Implement `expected_accepted_length(tree, accept_prob)`:

```python
def expected_accepted_length(tree: list[list[int]], accept_prob: list[float]) -> float:
    ...
```

`tree[i]` contains the children of node `i`. `accept_prob[i]` is the acceptance probability for node `i`. The root node `0` is not counted in the returned length, but its probability value may be provided and should be ignored.

Return the expected accepted length as a Python `float`.

Assumptions:

- The tree is valid and acyclic.
- Every probability is in the range $[0,1]$.
- The length of `accept_prob` matches the number of nodes.

## Example

```python
tree = [
    [1, 2],
    [3],
    [],
    []
]
accept_prob = [0.0, 0.5, 0.25, 0.8]

# Node 1 contributes 0.5
# Node 2 contributes 0.25
# Node 3 contributes 0.5 * 0.8 = 0.4
# Result: 1.15
```

## What the gate checks

The gate compares the implementation against an independently computed recursive reference model on several generated tree structures.

The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert x-\hat{x}\rVert_2}{\lVert x\rVert_2+10^{-12}}
$$

between the candidate result and the reference result must satisfy $\mathrm{rel\_err} \le 10^{-6}$.
