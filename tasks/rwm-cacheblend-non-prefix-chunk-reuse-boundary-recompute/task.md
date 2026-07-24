## Context

Large language model serving systems can reuse previous key-value cache entries when a new request contains text that has already been processed. A document is divided into fixed-size token chunks. Instead of requiring the matching text to be a prefix, a content-addressed cache can find matching chunks anywhere in the previous documents.

For a chunk $c = (t_1, t_2, \dots, t_k)$, the cache identity is based on its content hash:

$$
h(c) = \mathrm{hash}(t_1, t_2, \dots, t_k).
$$

A reusable chunk can copy its stored KV states, but attention dependencies at chunk boundaries may require recomputing selected tokens. This task models a simplified CacheBlend planner: reuse complete matching chunks and recompute boundary tokens around reused chunk seams.

Given a previous cache containing tokenized documents and a new document, the planner should identify every new chunk whose contents appear in the cache. Chunk positions do not need to match. Multiple identical chunks reuse the earliest matching cached occurrence.

The boundary recomputation rule is deterministic:

$$
R = \bigcup_{(i,\_,\_)\in U} \{i \cdot s,\ i \cdot s + s - 1\},
$$

where $U$ is the set of reused new chunk indices and $s$ is the chunk size. Token positions outside the document length are ignored.

## Task

Implement `cacheblend_plan(cached_docs, new_doc, chunk_size)`.

Arguments:

- `cached_docs` is a list of token lists. Each token is an integer.
- `new_doc` is a token list for the incoming document.
- `chunk_size` is a positive integer.

The function must return a dictionary with exactly these keys:

```python
{
    "reuse": [
        [new_chunk_index, cached_doc_index, cached_chunk_index],
        ...
    ],
    "recompute": [token_index, ...]
}
```

The `reuse` list must be ordered by increasing `new_chunk_index`. The `recompute` list must contain unique sorted token indices.

A document is split into consecutive non-overlapping chunks. The final partial chunk is ignored because only complete chunks can have reusable KV cache entries.

## Example

```python
cached_docs = [
    [1, 2, 3, 4, 9, 9],
    [8, 7, 6, 5]
]

new_doc = [8, 7, 6, 5, 1, 2, 3, 4]

cacheblend_plan(cached_docs, new_doc, 4)

# {
#   "reuse": [
#     [0, 1, 0],
#     [1, 0, 0]
#   ],
#   "recompute": [0, 3, 4, 7]
# }
```

## What the gate checks

The gate builds several documents with overlapping chunks in different positions. It computes the expected plan using an independent content-hash oracle that scans all cached chunks, selects the earliest matching occurrence, and applies the boundary recomputation rule.

The `exact_match` metric must equal $1.0$. Prefix-only reuse, position-based matching, or missing boundary recomputation will fail on the crafted non-prefix overlap cases.
