## Context

Block-sparse attention kernels tile the score matrix into
`block_size x block_size` tiles and skip tiles that are provably fully
masked, to avoid wasted compute. For query block $i$ and key block $j$
(0-indexed, causal masking, $j \le i$ always required for any visibility):

- $j < i$: every key in block $j$ is strictly before every query in block
  $i$ — the tile is **fully visible**, no masking needed, compute it dense.
- $j > i$: every key in block $j$ is strictly after every query in block
  $i$ — the tile is **fully masked** (empty), safe to skip entirely.
- $j = i$: the **diagonal** tile. Within it, query row $r$ can see key
  column $c$ only if $c \le r$ — it is neither fully visible nor fully
  masked, it's **partial**. A partial tile still contributes real,
  non-skippable attention mass (in particular every query's attention to
  *itself*, always the closest key) — it must be computed with an
  elementwise causal mask, never skipped.

The kernel accumulates an online (numerically-stable, running-max) softmax
over key blocks $j = 0 \ldots i$ for each query block $i$:

$$
m^{\text{new}} = \max(m, \max_c \text{scores}_{:,c}), \qquad
p = \exp(\text{scores} - m^{\text{new}}),
$$

$$
\ell \leftarrow \ell \cdot e^{m - m^{\text{new}}} + \sum_c p_{:,c}, \qquad
\text{acc} \leftarrow \text{acc} \cdot e^{m - m^{\text{new}}} + p V_{\text{block}}, \qquad
m \leftarrow m^{\text{new}}
$$

with the final output $\text{acc} / \ell$ per query row.

## Task

`starter.py` contains a block-sparse causal self-attention kernel with a
bug: its inner loop over key blocks only visits `j` in `range(i)` —
**strictly before** the query block — so the diagonal tile ($j = i$) is
never visited at all. It was written to "skip empty tiles" but the check
that decides what's empty misclassifies the partial diagonal tile as
empty too, so every query silently loses all of its own block's attention
mass (including attending to itself).

Fix `block_sparse_causal_attention(q, k, v, block_size)` so the diagonal
tile is **computed, not skipped**, with the correct elementwise causal
mask ($c \le r$ within the tile) applied before folding it into the
running softmax:

```python
def block_sparse_causal_attention(q, k, v, block_size):
    ...
```

- `q, k, v`: `(n, d)` float64 arrays, `n` a multiple of `block_size`.
- Returns the `(n, d)` causal self-attention output
  $\mathrm{softmax}(QK^\top / \sqrt{d})V$ with future positions masked,
  computed via the block-tiled algorithm above (fully-visible tiles dense,
  fully-masked tiles skipped, the diagonal tile masked elementwise —
  never skipped).

## Example

```python

n, d, block_size = 8, 4, 4
rng = random.Random(0)
q = k = v = rng.normal(size=(n, d))

out = block_sparse_causal_attention(q, k, v, block_size)
# Row 0 must equal v[0] exactly: with only itself visible, softmax over a
# single score is 1.0, so out[0] == v[0]. The buggy kernel returns NaN for
# row 0 (and every other row of the first block), since it skips the only
# tile row 0 could ever attend to.
```

## What the gate checks

The grader builds several `(n, d, block_size)` scenarios from a seeded
Python generator (block sizes that tile the sequence evenly, multiple
blocks per sequence) and computes the reference output independently as
**dense** causal attention in float64:
`softmax(QK^T/sqrt(d) with future positions masked to -inf) @ V`, applied
directly to the whole `(n, n)` score matrix — never calling your block
kernel and never hardcoding an expected array.

`max_abs_err` is the worst-case elementwise absolute error between your
output and the dense oracle across all scenarios, and the gate requires
`<= 1e-5`. Dropping the diagonal tile leaves the first query block's rows
with zero total softmax weight (`NaN`, an instant, unambiguous fail) and
every later block missing its own closest, most-attended keys — both
comfortably fail the gate; only computing every tile the algorithm above
calls for (dense below the diagonal, masked on it, skipped above it)
reaches `1e-5`.
