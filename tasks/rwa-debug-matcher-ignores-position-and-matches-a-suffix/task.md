## Context

Prefix reuse is used by systems that cache work for repeated sequences. A matcher must only reuse a cached prefix when the query tokens are equal to the cached tokens at the same positions.

For two token sequences $c$ and $q$, the positional longest common prefix (LCP) length is the largest integer $k$ such that

$$
\forall i \in \{0, \dots, k-1\}, \quad c_i = q_i .
$$

A common bug is to check whether query tokens appear somewhere in the cached sequence and then report reuse. This confuses membership with positional equality. For example, a cache `[4, 7, 9, 2]` and query `[7, 9]` have matching token IDs, but their positional prefix reuse length is $0$ because `7` is not at position $0`.

## Task

Implement `prefix_reuse_length(cache_tokens, query_tokens)`.

The function receives two sequences of integer token IDs:

```python
def prefix_reuse_length(cache_tokens, query_tokens):
    ...
```

Return the number of tokens that can be reused from the beginning of the query. The returned value must be the positional LCP length.

Do not search for matching subsequences, suffixes, or individual token membership. Only the aligned positions from the start of both sequences count.

## Example

```python
cache = [10, 20, 30, 40]
query = [10, 20, 99]

prefix_reuse_length(cache, query)
# 2
```

The first two positions match, then `30 != 99`, so the reusable prefix length is `2`.

## What the gate checks

The gate builds several cache/query pairs and computes the expected result using a positional LCP oracle. The implementation passes only when its returned reuse length exactly matches the oracle for every case.

A matcher that finds a matching suffix or any matching subsequence will over-report reuse and fail cases where equal token IDs occur at different positions.
