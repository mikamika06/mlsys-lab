## Context

A paged KV-cache stores each sequence's keys/values as a chain of fixed-size
**blocks** of `block_size` tokens. When two sequences share an identical
leading run of tokens (e.g. the same system prompt, or two branches of the
same request), a naive cache duplicates the KV blocks for both sequences.
Production engines instead use **copy-on-write (COW)** block sharing: any
block that is *entirely* covered by the shared prefix is stored **once**
and both sequences hold a reference to it; only the diverging suffix gets
its own private blocks.

For two sequences of length $L_A$ and $L_B$ tokens with a shared prefix of
length $S$ (measured in tokens) and page size `block_size`:

$$
\text{blocks}(L) = \left\lceil \frac{L}{\text{block\_size}} \right\rceil,
\qquad
\text{shared\_blocks} = \left\lfloor \frac{S}{\text{block\_size}} \right\rfloor
$$

(a block only counts as shared if *every* token in it lies within the
shared prefix — a partially-shared block still needs its own private copy
in both sequences, so `shared_blocks` never exceeds `blocks(L_A)` or
`blocks(L_B)`).

$$
\text{duplicated} = \text{blocks}(L_A) + \text{blocks}(L_B), \qquad
\text{unique} = \text{duplicated} - \min(\text{shared\_blocks}, \text{blocks}(L_A), \text{blocks}(L_B))
$$

$$
\text{size\_ratio} = \frac{\text{duplicated}}{\text{unique}}
$$

Crucially, COW block sharing is a **memory** optimization only — it changes
how many physical blocks are stored, but each sequence's attention output
is still computed over its **own full** queries/keys/values, exactly as if
no sharing had happened. Sharing must never change the numbers that come
out of attention.

## Task

Implement `cow_prefix_attention`:

```python
def cow_prefix_attention(q_a, k_a, v_a, q_b, k_b, v_b, shared_prefix_len, block_size):
    ...
```

- `q_a, k_a, v_a`: arrays of shape `(L_A, d)` — per-token queries/keys/values
  for sequence A, float64.
- `q_b, k_b, v_b`: arrays of shape `(L_B, d)` for sequence B.
- `shared_prefix_len`: number of leading tokens A and B have in common
  ($S$ above).
- `block_size`: KV-cache page size in tokens (positive `int`).

Return `(size_ratio, out_a, out_b)`:

- `size_ratio` — the float defined above (`duplicated / unique` physical
  blocks).
- `out_a` — standard **causal** scaled dot-product self-attention output
  for sequence A: for each row $i$, attend only to keys/values at
  positions $j \le i$, using
  $\mathrm{softmax}(QK^\top / \sqrt{d})V$ with the upper-triangular future
  positions masked to $-\infty$ before the softmax. Shape `(L_A, d)`.
- `out_b` — the same computation for sequence B, shape `(L_B, d)`.

`out_a` and `out_b` must each be computed independently over their own
full `q, k, v` — the COW sharing described above must **not** change
either output.

## Example

```python

d = 4
q_a = k_a = v_a = [[float(i * d + j) for j in range(d)] for i in range(3)]
q_b = k_b = v_b = [[float(i * d + j) for j in range(d)] for i in range(3)]

ratio, out_a, out_b = cow_prefix_attention(q_a, k_a, v_a, q_b, k_b, v_b,
                                            shared_prefix_len=3, block_size=2)
# blocks(3) = ceil(3/2) = 2 each; shared_blocks = floor(3/2) = 1
# duplicated = 2 + 2 = 4; unique = 4 - 1 = 3; ratio = 4/3
# out_a and out_b are identical here since inputs are identical
```

## What the gate checks

The grader builds several `(q_a, k_a, v_a, q_b, k_b, v_b, shared_prefix_len,
block_size)` scenarios — varying sequence lengths, block sizes that divide
evenly and ones that leave a remainder, a shared prefix of zero, a shared
prefix equal to the shorter sequence's full length, and seeded-random
query/key/value arrays — and computes the reference `size_ratio` and both
causal attention outputs independently in Python (float64 throughout: block
counts from `ceil`/`floor` on the lengths, attention via a masked
`softmax(QK^T/sqrt(d))V`), never calling your function or hardcoding an
expected value.

Two gates apply: `ratio_err` is the worst-case absolute difference between
your `size_ratio` and the oracle's across all scenarios (must be
`<= 1e-9`), and `max_abs_err` is the worst-case elementwise absolute error
between your `out_a`/`out_b` and the oracle's attention outputs across all
scenarios (must be `<= 1e-5`). Counting every block as private (ignoring
sharing), sharing a block that is only partially covered by the prefix, or
computing attention over a merged/deduplicated KV instead of each
sequence's own full KV will all fail one of the two gates.
