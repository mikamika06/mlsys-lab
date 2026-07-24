## Context

Prefix-cache engines (SGLang's RadixAttention, vLLM's automatic prefix
cache) share KV cache across requests with a common token prefix by storing
generated sequences in a **radix tree**: a compressed prefix tree where
every edge holds a *run* of tokens (not just one), and a node only branches
once two inserted sequences actually diverge.

Inserting a token sequence $s$ starting at node $v$ with an existing child
edge $e$ (whose first token matches $s$'s first token) requires comparing
$s$ against $e$'s tokens position by position, up to their common length,
to find the longest common prefix length $i$:

$$
i = \max\{\, j : s_{0:j} = e_{0:j} \,\}.
$$

Three cases:

- $i = |e|$ (the whole edge matches): the edge is fully consumed: recurse
  into the child with the remaining suffix $s_{i:}$.
- $i < |e|$ and $i = |s|$: $s$ ends exactly inside the edge — **split** the
  edge into $e_{0:i}$ (kept on the original edge, now pointing to a new
  intermediate node) and $e_{i:}$ (moved to a new edge below the
  intermediate node, pointing to the original child). No new leaf is
  needed; $s$'s path ends at the new intermediate node.
- $i < |e|$ and $i < |s|$: both diverge — same split as above, **plus** a
  new leaf edge $s_{i:}$ hangs off the new intermediate node.

If no existing child edge shares $s$'s first token, a brand-new leaf edge
$s$ is added directly.

## Task

Implement `build_radix_tree`:

```python
def build_radix_tree(sequences: list) -> list:
    ...
```

- `sequences` — a list of token sequences (each a list/tuple of ints),
  inserted **in the given order** into an initially empty radix tree
  rooted at the empty path.

Return a canonical serialization of every edge in the resulting tree: a
sorted list of `(parent_path, edge_tokens)` pairs, both as tuples of ints,
where `parent_path` is the full token path from the root to the edge's
parent node (`()` for edges hanging directly off the root) and
`edge_tokens` is the tuple of tokens on that edge.

## Example

```python
build_radix_tree([[1, 2, 3, 4], [1, 2, 5, 6]])
# ->
# [((), (1, 2)), ((1, 2), (3, 4)), ((1, 2), (5, 6))]
```

The second insert shares only `[1, 2]` with the first before diverging, so
the original single edge `(1,2,3,4)` splits into `(1,2)` followed by a
branch into `(3,4)` and `(5,6)`.

## What the gate checks

The fixture holds several independent insertion **runs** — hand-picked
cases (a basic split, re-inserting an already-present prefix as a no-op,
splitting exactly at an edge's end with no extra leaf, fully disjoint
sequences, duplicate inserts, a three-way split under an already-split
node, and nested prefixes inserted both shortest-first and longest-first)
plus several random small-vocabulary runs that force frequent branching.
For every run, the grader builds the reference radix tree with the same
algorithm and compares the sorted `(parent_path, edge_tokens)`
serialization to yours exactly (`exact_match == 1.0`). Forgetting the
no-extra-leaf case when $i = |s| < |e|$, mismatching which node keeps
which edge remainder after a split, or not treating a re-inserted prefix
as a no-op will diverge from the reference on at least one run.
