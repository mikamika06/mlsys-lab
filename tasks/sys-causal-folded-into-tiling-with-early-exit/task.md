## Context

Dense causal attention computes a full `(seq, seq)` score matrix and
masks out the upper triangle (`key_idx > query_idx`) before the
softmax — but for a *tiled* implementation (the kind flash-attention
style kernels use to keep blocks in fast memory), a whole
query-tile/KV-tile pair can be skipped **without computing any scores
for it at all** whenever every key in that KV tile is strictly in the
future of every query in that query tile:

$$
\text{skip}(q_i, k_j) \iff k_j^{\text{start}} > q_i^{\text{end}}
$$

(using 0-based, inclusive `end` indices — the KV tile starts after the
query tile's last position, so causally every entry in it would be
masked anyway). Tiles that aren't skipped still need a causal mask
applied to the entries *within* the block (`key_idx > query_idx`,
using each tile's true global row/column indices) — this correctly
handles both "fully in the past" tiles (mask has no effect) and
"diagonal" tiles (partially masked) with the same code path.

## Task

Implement `tiled_causal_attention`:

```python
def tiled_causal_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], tile_q: int, tile_kv: int, on_tile=None) -> list[list[float]]:
    ...
```

- `Q`, `K`, `V`: `(seq_len, d)` `float64` (single head, no batch).
  `seq_len` is divisible by both `tile_q` and `tile_kv`.
- `tile_q`, `tile_kv`: query-tile and KV-tile sizes.
- `on_tile`: optional callable. If provided, call `on_tile(qi, kj)`
  **exactly once**, right before scoring, for every query-tile index
  `qi` / KV-tile index `kj` pair that is **not** skipped by the rule
  above — and never call it for a skipped (fully-future) pair.

Algorithm: initialize an `(seq_len, seq_len)` score buffer to `-inf`.
For every query tile `qi` and KV tile `kj`: if
`kj*tile_kv > (qi+1)*tile_q - 1`, skip it entirely (no `on_tile` call,
no scoring). Otherwise call `on_tile(qi, kj)` (if given), compute
`block = Q[q_tile] @ K[kv_tile].T / sqrt(d)`, mask `block` where the
tile's global key index exceeds its global query index (set to
`-inf`), and write it into the score buffer at
`[q_tile_rows, kv_tile_cols]`. After all tiles: row-wise softmax the
full buffer, then `out = softmax(scores) @ V`.

## Example

```python
seq, d = 8, 4
import random; random.seed(0); Q = K = V = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq)]
visited = []
out = tiled_causal_attention(Q, K, V, tile_q=4, tile_kv=4, on_tile=lambda qi, kj: visited.append((qi, kj)))
# visited == [(0, 0), (1, 0), (1, 1)] -- query tile 0 only ever needs KV
# tile 0 (KV tile 1 starts at position 4, strictly after query tile 0's
# last position 3, so it's skipped); query tile 1 needs both KV tiles.
```

## What the gate checks

The grader builds several seeded `(Q, K, V, tile_q, tile_kv)`
configurations. It computes a reference output with plain dense causal
attention (full score matrix, upper-triangle masked, single softmax)
and separately computes, from the skip rule itself, the exact set of
`(qi, kj)` pairs a correct tiled implementation should visit.

`max_abs_err` is the worst-case max elementwise absolute difference
between your `out` and the dense reference, across all configurations
(must be `<= 1e-5`) — a wrong mask or a wrong skip boundary changes
the actual attention output, not just its cost.
`tile_visit_exact_match` is `1.0` only if the *set* of `(qi, kj)` pairs
your implementation passed to `on_tile` matches the expected skip
pattern exactly on every configuration (must equal `1.0`) — this
catches a solution that gets the numerically correct answer by
computing (and masking) every tile including the fully-future ones,
i.e. one that never actually exploits the early exit.
