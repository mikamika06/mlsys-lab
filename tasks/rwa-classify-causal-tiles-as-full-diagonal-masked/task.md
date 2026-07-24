## Context

In a causal (autoregressive) attention kernel such as FlashAttention-2, the attention matrix is lower-triangular: position $q$ can only attend to positions $k \le q$. When the computation is tiled into blocks of size $B$, each $(i, j)$ tile spans query rows $[i \cdot B,\, (i+1) B)$ and key columns $[j \cdot B,\, (j+1) B)$.

A tile can be classified without inspecting every element:

- **full** — all $(q, k)$ pairs satisfy $k \le q$, i.e. the entire tile is attended. This happens when the last key index is $\le$ the first query index: $(j+1)B - 1 \le iB$, i.e. $j < i$.
- **empty** — no $(q, k)$ pair satisfies $k \le q$, i.e. the tile is entirely masked. This happens when the first key index exceeds the last query index: $jB > (i+1)B - 1$, i.e. $j > i$.
- **diagonal** — the tile straddles the causal boundary ($i = j$): some elements are attended and some are masked.

Formally for a sequence of length $L$ divided into $N = L / B$ blocks:

$$\text{label}(i, j) = \begin{cases} \texttt{"full"} & j < i \\ \texttt{"diagonal"} & j = i \\ \texttt{"empty"} & j > i \end{cases}$$

This classification is done once per block pair to decide whether to compute the full tile, skip it, or apply a fine-grained mask.

## Task

Implement `classify_causal_tiles(seq_len, block_size)`:

```python
def classify_causal_tiles(seq_len, block_size):
    ...
```

- `seq_len` — total sequence length (guaranteed multiple of `block_size`).
- `block_size` — tile size in tokens.
- Returns a 2-D list of shape `(num_blocks, num_blocks)` where `num_blocks = seq_len // block_size`. Each entry is `"full"`, `"empty"`, or `"diagonal"`.

## Example

```python
result = classify_causal_tiles(4, 2)
# num_blocks = 2
# result[0][0]: i=0, j=0 -> diagonal
# result[0][1]: i=0, j=1 -> empty
# result[1][0]: i=1, j=0 -> full
# result[1][1]: i=1, j=1 -> diagonal
# => [["diagonal", "empty"], ["full", "diagonal"]]
```

## What the gate checks

`exact_match`: The grader exhaustively evaluates `k <= q` for every $(q, k)$ element in each tile using NumPy, labels each tile, and checks the student's returned grid matches exactly across multiple `(seq_len, block_size)` combinations.
