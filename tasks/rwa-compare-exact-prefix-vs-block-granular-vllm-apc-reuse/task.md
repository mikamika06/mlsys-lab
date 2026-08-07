## Context

Two production prefix-caching schemes reuse previously computed KV-cache
entries for a new request, but at different granularity:

- **RadixAttention** (SGLang) indexes cached sequences in a radix tree over
  individual tokens, so it can reuse a shared prefix of **any** length —
  exactly up to the first token where the new request diverges from every
  cached sequence.
- **vLLM's Automatic Prefix Caching (APC)** stores the KV cache in
  fixed-size blocks of `block_size` tokens and can only reuse **whole**
  blocks — a block is only reusable if *every* token in it matches, so any
  partial match inside the last block is thrown away.

For a query token sequence $q$ and a cache of previously seen sequences
$\{c^{(1)}, \dots, c^{(k)}\}$, define the exact reuse length as the longest
prefix $q$ shares with *any* cached sequence:

$$
\mathrm{lcp}(q, c) = \max\{\, j \ge 0 : q_0 \dots q_{j-1} = c_0 \dots c_{j-1} \,\}
$$

$$
L_{\text{exact}}(q) = \max_i \; \mathrm{lcp}(q, c^{(i)})
$$

Block-granular reuse rounds that down to the nearest whole block:

$$
L_{\text{block}}(q) = \left\lfloor \frac{L_{\text{exact}}(q)}{\text{block\_size}} \right\rfloor \cdot \text{block\_size}
$$

The gap $L_{\text{exact}}(q) - L_{\text{block}}(q)$ (up to
$\text{block\_size} - 1$ tokens) is exactly the KV-cache compute that APC
redoes but RadixAttention would have reused.

## Task

Implement `prefix_reuse_lengths`:

```python
def prefix_reuse_lengths(cache: list[list[int]], queries: list[list[int]], block_size: int) -> list[tuple[int, int]]:
    ...
```

- `cache` is a list of previously-seen token-id sequences (each a list of
  `int`s; may be empty).
- `queries` is a list of new token-id sequences to score against `cache`.
- `block_size` is a positive `int`.
- For each query, return a `(exact_reuse, block_reuse)` pair:
  `exact_reuse` is $L_{\text{exact}}(q)$ as defined above (the longest
  prefix shared with *any* single cache entry — not a mix of several), and
  `block_reuse` is $L_{\text{block}}(q)$, the same value rounded down to
  the nearest multiple of `block_size`.
- If `cache` is empty, or no cache entry shares even a first token with a
  query, both values are `0`.

## Example

```python
cache = [list(range(37)) + [999, 999]]   # a 37-token shared prefix
queries = [list(range(37)) + [500, 600]]  # diverges at token 37

prefix_reuse_lengths(cache, queries, block_size=16)
# [(37, 32)]   -- RadixAttention reuses 37 tokens; vLLM APC only 2 whole
#                 16-token blocks (32 tokens) -- 5 tokens recomputed for
#                 nothing.
```

## What the gate checks

The grader builds several `(cache, queries, block_size)` scenarios —
overlapping prefixes of varying lengths, multiple cache entries where the
best match isn't the first one, a query shorter than its best-matching
cache entry, and queries with no match at all — and computes the reference
`(exact_reuse, block_reuse)` for every query with an independent Python
implementation: pad each `(query, cache-entry)` pair to equal length,
compare element-wise, and locate the first mismatch with
`max`/`all()` on the boolean equality list (never calling your
function, never hardcoding an expected answer).

`exact_match` is the fraction of queries where **both** returned values
equal the oracle's exactly; the gate requires `1.0`. Reusing the maximum
over the *wrong* cache entry, mixing up exact vs. block-rounded values, or
rounding with the wrong block size all produce a mismatch somewhere in the
scenario set.
