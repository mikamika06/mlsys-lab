## Context

As a server processes a trace of requests one at a time, each new
sequence can reuse KV cache from **any** sequence processed so far.
Two caching schemes disagree on how much of a shared prefix actually
counts as reusable:

- **Radix (token-level)** reuse: the longest prefix shared with *any*
  previously-seen sequence, down to the exact token — SGLang's
  radix-tree cache works this way.
- **Block-level** reuse: only *whole* fixed-size blocks of `block_size`
  tokens can be reused — vLLM's Automatic Prefix Caching (APC) works this
  way, since it hashes and shares memory in block-sized pages.

For sequence $i$ (with sequences $0, \dots, i-1$ already cached):

$$
\mathrm{lcp}(q, c) = \max\{\, j \ge 0 : q_0\dots q_{j-1} = c_0 \dots c_{j-1} \,\}
$$

$$
\text{radix}_i = \max_{j < i} \mathrm{lcp}(\text{seqs}_i, \text{seqs}_j), \qquad
\text{block}_i = \left\lfloor \frac{\text{radix}_i}{\text{block\_size}} \right\rfloor \cdot \text{block\_size}
$$

(both are $0$ for $i = 0$, and $0$ for any $i$ with no prior match).
Because $\lfloor x/B \rfloor \cdot B$ never exceeds $x$, block-level
reuse can never beat radix reuse on any single sequence — or in total.

## Task

Implement `compute_reuse_savings`:

```python
def compute_reuse_savings(seqs: list[list[int]], block_size: int) -> tuple[int, int]:
    ...
```

- `seqs`: a list of `N` token-id sequences (each a list of `int`s),
  processed strictly in the given order. Sequence `i` may reuse from any
  of `seqs[0], ..., seqs[i-1]` (already "cached" by the time `i` arrives)
  — never from itself or from a later sequence.
- `block_size`: a positive `int`.

Return `(radix_saved_tokens, block_saved_tokens)`: the totals of
$\text{radix}_i$ and $\text{block}_i$ (as defined above) **summed over
every sequence** $i = 0, \dots, N-1$ in the trace.

## Example

```python
seqs = [
    list(range(50)),                       # nothing to reuse from (first)
    list(range(37)) + [999, 999],          # shares 37 tokens with seq 0
    list(range(20)) + [500],               # shares 20 with seq 0, 20 with seq 1
]
compute_reuse_savings(seqs, block_size=16)
# radix: 0 + 37 + 20 = 57
# block: 0 + floor(37/16)*16 (=32) + floor(20/16)*16 (=16) = 48
# -> (57, 48)
```

## What the gate checks

The grader builds several `(seqs, block_size)` traces from a seeded
NumPy generator — including traces engineered to diverge partway through
a block (so token-level and block-level reuse clearly differ), a trace
where the best-matching prior entry isn't the immediately preceding one,
and traces with no shared prefixes at all — and computes the reference
`(radix_saved_tokens, block_saved_tokens)` independently in Python by the
definitions above, never calling your function or hardcoding an expected
value.

`exact_match` is the fraction of scenarios where **both** totals match
the oracle's exactly, and the gate requires `1.0`; the grader additionally
sanity-checks `radix_saved_tokens >= block_saved_tokens` (a violation on
either side — yours or the oracle's — signals a broken scenario or
solution). Comparing a sequence only against its immediate predecessor
instead of every prior sequence, letting a sequence match against
itself or a later sequence, or rounding radix reuse with `round`/`ceil`
instead of `floor` will all produce a mismatch on at least one trace.
