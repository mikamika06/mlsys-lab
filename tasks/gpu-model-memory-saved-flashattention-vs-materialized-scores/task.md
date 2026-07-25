## Context

Standard attention computes the full score matrix
$S = QK^T \in \mathbb{R}^{\text{seq\_len} \times \text{seq\_len}}$ and
materializes it — every query against every key, all at once, resident
in memory before softmax and the weighted sum even start. That's a
peak intermediate footprint that grows with the *square* of the sequence
length.

FlashAttention never builds $S$ at all. It processes keys in
`block_size`-sized chunks, computing only a `block_size x block_size`
tile of scores at a time, folding each tile's contribution into a
running (online-softmax) output, and discarding the tile before the next
one is computed. Its peak intermediate footprint is fixed by
`block_size` alone — it never depends on `seq_len`.

## Task

Implement

```cpp
__global__ void flash_vs_materialized_ratio(float* out, const float* seq_len, const float* block_size,
                                             const float* bytes_per_elem, int n);
```

For every scenario `i` in `[0, n)`:

$$
\text{materialized\_bytes} = \text{seq\_len}[i]^2 \times \text{bytes\_per\_elem}[i]
$$
$$
\text{flash\_bytes} = \text{block\_size}[i]^2 \times \text{bytes\_per\_elem}[i]
$$
$$
\text{out}[i] = \frac{\text{materialized\_bytes}}{\text{flash\_bytes}}
$$

## Example

`seq_len=2048, block_size=64` (independent of `bytes_per_elem`, which
cancels in the ratio): `materialized_bytes / flash_bytes =
(2048/64)^2 = 32^2 = 1024` — FlashAttention's peak intermediate memory
is 1024x smaller, purely from tiling the score computation instead of
materializing all of it at once.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it
on 5 fixed `(seq_len, block_size, bytes_per_elem)` scenarios, comparing
`out` against a reference computed directly with numpy (never a
hardcoded list of ratios). It requires

$$
\mathrm{max\_abs\_err} \le 10^{-3}
$$

On this fixture the ratios span `16` up to `1024`, always exactly
$(\text{seq\_len}/\text{block\_size})^2$ — note `bytes\_per\_elem` always
cancels out of the ratio, since it multiplies both the numerator and
denominator identically; it's included in the signature because a
learner who forgets it cancels (and tries to special-case it) would
otherwise be tempted to introduce a bug that isn't actually there.
Computing a *linear* ratio (`seq_len/block_size`, forgetting the square)
instead of the quadratic one gets every scenario wrong by exactly a
factor of `seq_len/block_size` — e.g. `32` instead of `1024` on the
`seq_len=2048, block_size=64` case.
