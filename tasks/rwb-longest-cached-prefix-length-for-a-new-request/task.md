## Context

Production block-level prefix caches (vLLM's Automatic Prefix Caching,
SGLang's RadixAttention) identify a cached block not by its own content
alone, but by a **chained hash**: block $b$'s cache key folds together
the *previous* block's key with this block's own tokens, so it encodes
the entire prefix that led to it, not just $b$ in isolation:

$$
H_0 = \text{block\_hash}(0,\ \text{tokens}[0{:}B]), \qquad
H_b = \text{block\_hash}\big(H_{b-1},\ \text{tokens}[bB{:}(b{+}1)B]\big)
$$

using the rolling hash

$$
\text{block\_hash}(p, t_1,\dots,t_B) = \Big(\cdots\big((p\cdot k + t_1 + 1)\cdot k + t_2 + 1\big)\cdots \cdot k + t_B + 1\Big) \bmod m
$$

with $k = 1{,}000{,}003$ and $m = 2^{61}-1$. This means a new request can
only reuse block $b$ from the cache if **every** block before it also
matched exactly — reusing block 2 requires blocks 0 and 1 to have
matched too, because $H_2$ depends on $H_1$ which depends on $H_0$. Two
requests whose block 1 has identical *tokens* but arrived via a
*different* block 0 get completely different (and essentially
never-colliding) hashes for block 1.

## Task

Implement `longest_cached_prefix_blocks`:

```python
def longest_cached_prefix_blocks(cached_hashes: set, new_tokens: list, block_size: int) -> int:
    ...
```

- `cached_hashes`: a set of int chain hashes currently resident in the
  cache (as produced by the formula above, applied to some other
  requests' full blocks).
- `new_tokens`: list of int token ids for the new request.
- `block_size` (`B`): positive int.
- Compute `new_tokens`'s own chain hashes, block 0's parent is `0`.
  Walk blocks in order and count how many **consecutive leading** blocks
  hit `cached_hashes`, stopping at the first miss. A trailing block with
  fewer than `B` tokens can never hit (block caches only store complete
  blocks) and also ends the walk.
- Return that count (an int between `0` and `len(new_tokens) // B`).

## Example

```python
B = 4
new_tokens = [1, 2, 3, 4,  5, 6, 7, 8,  55, 66, 77, 88,  9, 9, 9, 9,  42]
# block 0 = [1,2,3,4], block 1 = [5,6,7,8], block 2 = [55,66,77,88], ...

# Suppose the cache holds another request's blocks
#   [1,2,3,4], [5,6,7,8], [100,101,102,103], [9,9,9,9]
# (chained). new_tokens' block 0 and block 1 are byte-identical AND in
# the same chain position, so they hit. block 2 differs in content from
# the cached request's block 2, so its chain hash misses -- even though
# some UNRELATED cached request elsewhere might happen to contain the
# raw tokens [55,66,77,88] at a different chain position, that does not
# count, because its hash was built from a different parent.

longest_cached_prefix_blocks(cached_hashes, new_tokens, B)  # 2
```

## What the gate checks

The grader loads a committed fixture built exactly this way — including
a **decoy** cached block whose raw content matches the new request's
missing block but at a different chain position — plus several
additional seeded synthetic cache/request pairs with varying
`block_size` and shared-prefix lengths, and computes the expected hit
count with an independent implementation of the same chained-hash walk
— never calling your function, never hardcoding an expected value.

`exact_match` is the fraction of cases where your returned integer
exactly equals the oracle's, and must be `1.0`. Hashing each block's raw
content alone (ignoring the parent hash) will pass the fixture's first
two blocks but incorrectly report a hit at block 2 because of the decoy;
using the wrong parent for block 0, or not stopping at the first miss,
will also produce a wrong count.
