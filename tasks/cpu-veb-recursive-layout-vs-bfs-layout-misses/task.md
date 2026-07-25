## Context

The standard array layout for a binary tree — root at index `0`, left
child of `p` at `2p+1`, right at `2p+2` (BFS/level order) — is compact
and simple, but a root-to-leaf search on it has poor locality: node
indices roughly *double* every level, so by a few levels down,
consecutive nodes on the search path sit far apart in the array, almost
always in different cache lines. A search of a height-`H` tree touches
close to `H` different lines no matter how big a line is.

The **van Emde Boas (vEB) layout** fixes this without knowing the cache
line size in advance (that's the "cache-oblivious" part). It recursively
splits a height-`h` tree into a top half of height `ceil(h/2)` and, below
it, `2^{ceil(h/2)}` bottom subtrees of height `floor(h/2)` — laying the
top subtree's nodes *contiguously first*, then each bottom subtree's
nodes contiguously after it, each one recursively laid out the same way.
The effect: any subtree small enough to fit in a cache line ends up
stored *inside* one, so a root-to-leaf search only "jumps" to a new
region of memory `O(log_B N)` times, where `B` is how many nodes fit in
a line — provably fewer regions than BFS layout needs, for any `B`.

## Task

Implement

```cpp
int veb_pos(const bool* path, int depth, int H);
```

`path[0..depth)` is a node's sequence of left(`false`)/right(`true`)
choices from the root of a complete, height-`H` binary tree
(`depth == 0` is the root itself). Recursively, on a subtree of height
`h` whose block starts at array offset `base`: if `h <= 1` the subtree is
one node, at `base`. Otherwise split `h1 = ceil(h/2)` (top) and
`h2 = h - h1` (bottom): if the node's `depth` is `< h1`, it's inside the
top subtree — recurse into height `h1`, same `base`. Otherwise, the
first `h1` steps of the path (from where this subtree's path starts)
pick which of the top subtree's `2^{h1}` leaves this node descends from
(`leaf_index`, treating those bits as a binary number); recurse into
that bottom subtree, height `h2`, based at
`base + (2^{h1} - 1) + leaf_index * (2^{h2} - 1)`, with the remaining
path bits. Return the node's final flat array slot.

## Example

Height `H = 4` (15 nodes): the root subtree splits into `h1 = 2` (top, 3
nodes: slots `0, 1, 2`) and 4 bottom subtrees of `h2 = 2` (3 nodes each).
The node reached by `path = [false, true, false]` (L, R, L): the first
`h1 = 2` bits (`L, R`) give `leaf_index = 1` (binary `01`), so it's in
the bottom subtree based at `3 + 1*3 = 6`. That bottom subtree (height 2)
recurses the same way on its own remaining path bit (`L`): its own
`h1 = 1`, top slot `6`, and since the remaining depth (`1`) isn't `< 1`,
it lands in ITS bottom half instead, at `6 + 1 + 0*1 = 7` — the node's
final slot is `7`.

## What the gate checks

The driver builds a height-`H=8` tree (255 nodes, 8-byte records,
64-byte lines — 8 nodes per line) and, for 3 fixed root-to-leaf paths
(all-left, all-right, alternating), walks every prefix of the path
(depths `0` through `7`) computing each node's vEB slot (your function)
and its fixed BFS/heap-array slot (the harness's own baseline), grouping
each into real 64-byte lines and counting the distinct lines touched
under each layout. It prints all 6 counts. The grader compiles
`solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{all 6 printed line counts match the reference}
$$

The reference prints
`leftmost: veb=3 bfs=5 | rightmost: veb=4 bfs=6 | zigzag: veb=3 bfs=6` —
vEB strictly beats BFS on every search. A stub that always returns slot
`0` collapses every node onto the same line (`veb=1` for all three
searches), which doesn't match the reference's real vEB counts and fails
the gate — while the `bfs` numbers, fixed by the harness, still match.
