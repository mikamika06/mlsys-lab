## Context

Naive attention computes the full `S x S` score matrix
`softmax(QK^T/sqrt(d))` before ever touching `V` — for long sequences
that matrix alone can dwarf the model itself. FlashAttention's core
trick is to never materialize it: stream over the keys/values, and keep
a *running* softmax normalizer that gets corrected on the fly every time
a bigger score is seen, using the identity

$$\text{softmax}(x)_j = \frac{e^{x_j - m}}{\sum_k e^{x_k - m}}$$

for *any* constant $m$ — so the running max can be updated mid-stream by
rescaling everything accumulated so far by $e^{m_{\text{old}} - m_{\text{new}}}$.

## Task

Implement

```c
__global__ void flash_attn(float* out, const float* Q, const float* K, const float* V, int S, int d, int bk)
```

One thread per query row (`i = threadIdx.x`). For row `i`, stream over
all `S` keys in chunks of `bk` (or one at a time — the online-softmax
math is identical either way), maintaining:

- `m` — the running max score seen so far (init `-1e30`)
- `l` — the running softmax denominator (init `0`)
- the running UNNORMALIZED output, stored directly in
  `out[i*d .. i*d+d)` (init to `0`)

For each key `j`: compute `s = (Q[i] . K[j]) / sqrt(d)`, `new_m =
max(m, s)`, rescale `l` and every entry of `out[i*d..i*d+d)` by
`exp(m - new_m)`, then add `exp(s - new_m)` to `l` and
`exp(s - new_m) * V[j]` to `out[i*d..i*d+d)`, and set `m = new_m`. After
all keys, divide `out[i*d..i*d+d)` by the final `l`.

## Example

For `S=8` keys and `d=4`, the score matrix would be `8x8` if
materialized — this kernel never allocates it: at any instant it holds
only `m`, `l`, and one `d`-length running output per row.

## What the gate checks

`max_abs_err` between the kernel's output and a numpy
`softmax(QK^T/sqrt(d))V` oracle, for one fixed random `Q`, `K`, `V`
(seed 3). Getting the rescale-on-new-max step wrong, forgetting to divide
by `l` at the end, or skipping the running-max correction entirely (plain
un-normalized exponentials, which overflow/misweight relative to the
true softmax) all produce visibly wrong output; an empty starter leaves
`out` at its initial zeros and fails outright (`max_abs_err ≈ 0.358`
against the real answer here).
