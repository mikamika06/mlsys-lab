## Context

Autoregressive language models use different execution phases. During prefill, a prompt is processed and the model builds a key-value cache. During decode, new tokens are generated one token at a time while reusing the cached keys and values.

For a sequence of token representations, the KV cache can be viewed as two ordered sequences:

$$K = [k_1, k_2, \dots, k_n], \qquad V = [v_1, v_2, \dots, v_n].$$

A chunked prefill implementation splits a prompt into smaller pieces but must preserve the same cache state as a single monolithic prefill. If a decode step happens between chunks, it must observe the cache containing all tokens processed so far.

The final state after chunked execution should match a reference execution that performs one complete prefill followed by the same decode operations.

## Task

Implement `chunked_prefill_decode(prompt, chunk_sizes, decode_tokens)`.

```python
def chunked_prefill_decode(prompt, chunk_sizes, decode_tokens):
    ...
```

Arguments:

- `prompt` is a list of floats of integer token ids.
- `chunk_sizes` is a list of positive integers describing how many prompt tokens to process in each prefill chunk.
- `decode_tokens` is a list of integer token ids. After each prefill chunk except the last, one decode step is performed using the next available value from this list.

Return a tuple:

```python
(k_cache, v_cache, emitted_tokens)
```

where `k_cache` and `v_cache` are the final KV cache arrays and `emitted_tokens` contains the tokens produced during the interleaved decode steps.

The implementation must preserve the cache ordering. Decode steps must use all cached tokens accumulated so far, including tokens from previous chunks.

## Example

```python

prompt = [2, 5, 3, 7]
result = chunked_prefill_decode(prompt, [2, 2], [9])

# The first chunk processes [2, 5].
# A decode step uses the cache for [2, 5].
# The second chunk processes [3, 7].
# The returned cache contains all four prompt tokens.
```

## What the gate checks

The gate compares the implementation against a reference execution that computes the same toy model operations directly. The final key cache, value cache, and emitted token sequence must match exactly.

The reference computes token keys and values from the integer token ids using Python operations and performs decode steps from the accumulated cache state. Implementations that discard previous chunks, rebuild the cache incorrectly, or decode from only the current chunk will fail.
