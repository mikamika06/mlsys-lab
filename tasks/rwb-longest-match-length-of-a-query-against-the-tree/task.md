## Context

A **trie** (prefix tree) is a rooted tree where each edge carries a token label.
A common operation in routing, autocomplete, and tokenization pipelines is the
**longest prefix match**: given a query sequence $q = (q_0, q_1, \dots, q_{m-1})$,
walk the tree from the root following edges whose labels equal successive query
tokens.  Stop at the first token $q_k$ that has no matching outgoing edge, or when
the query is exhausted.  The **longest match length** is the count of successfully
matched tokens:

$$L = \max\!\Big\{k \;\Big|\; \forall\; 0 \le i < k :\; q_i \in
\text{children}(\text{node}_i)\Big\}$$

where $\text{node}_0$ is the root and
$\text{node}_{i+1} = \text{children}(\text{node}_i)[q_i]$.

In this task the tree is represented as a **nested Python dict**: each dict maps a
token (string key) to the child node (another dict).  A leaf is an empty dict `{}`.
The root itself is a dict of top-level children.

## Task

Implement:

```python
def longest_match(tree: dict, query: list) -> int:
    """Return the longest prefix-match length of query against tree."""
    ...
```

`tree` is a nested dict as described above.  `query` is a list of string tokens.
Return the number of consecutive tokens from the start of `query` that can be
matched by walking edges in the tree.  Return `0` if the very first token has no
matching edge, or if `query` is empty.

Do **not** mutate the tree.

## Example

```python
tree = {
    "a": {
        "b": {"c": {}},
        "d": {}
    },
    "x": {}
}

longest_match(tree, ["a", "b", "c"])  # → 3
longest_match(tree, ["a", "b", "z"])  # → 2  (no "z" child under "b")
longest_match(tree, ["a"])             # → 1
longest_match(tree, ["y"])            # → 0  (no "y" child at root)
longest_match(tree, [])               # → 0
```

## What the gate checks

One gate: `exact_match`.  A non-trivial tree with multiple branches of varying
depth is combined with 20 diverse queries.  The grader recomputes the correct
answer for every query by walking the reference tree itself (its own oracle
implementation) and compares each result to the learner's output.  All 20 must
match for a score of $1.0$; any single mismatch yields $0.0$.
