## Context

A weight tensor compressed with **N:M structured sparsity** (keep $N$
of every $M$ consecutive weights, density $d = N/M$) plus low-bit
quantization of the survivors costs more than "just" $d \cdot
\text{bits}$ per original weight — real hardware formats (e.g. NVIDIA
Sparse Tensor Cores) pay two kinds of metadata on top:

1. **Position index**: each of the $N$ kept weights in a block needs to
   record *which* of the $M$ slots it came from, using a fixed
   $\lceil \log_2 M\rceil$-bit index (not an entropy-optimal code —
   real hardware formats use this simpler fixed-width encoding). Since
   a fraction $d$ of original weights are kept, this costs
   $d\cdot\lceil\log_2 M\rceil$ bits per **original** weight.
2. **Scale storage**: the kept weights are quantized in groups of
   `group_size` **original** weight positions, each group needing one
   stored scale of `scale_bits` bits — $\text{scale\_bits}/\text{group\_size}$
   bits per original weight, regardless of density.

Put together, the effective bits-per-weight of the whole compressed
representation (amortized over the *original*, dense parameter count)
is

$$
\text{bpw} = \underbrace{d\cdot\text{bits}}_{\text{payload}} +
\underbrace{d\cdot\lceil\log_2 M\rceil}_{\text{index overhead}} +
\underbrace{\frac{\text{scale\_bits}}{\text{group\_size}}}_{\text{scale overhead}}, \qquad d = \frac{N}{M}
$$

## Task

Implement `effective_bits_per_weight`:

```python
def effective_bits_per_weight(N: int, M: int, bits: int, group_size: int, scale_bits: float = 16.0) -> float:
    ...
```

- `N`, `M`: N:M sparsity pattern (keep `N` of every `M`, `1 <= N <= M`).
- `bits`: quantizer bit width applied to each kept weight.
- `group_size`: number of **original** (dense) weight positions sharing one stored scale.
- `scale_bits`: bits per stored scale value (default `16.0`, fp16).

Compute and return `bpw` using the formula above:
`density = N / M`, `index_bits = density * ceil(log2(M))`,
`scale_overhead = scale_bits / group_size`,
`bpw = density * bits + index_bits + scale_overhead`.

## Example

```python
# classic 2:4 sparsity, int4 kept weights, fp16 scale per 64 original weights
bpw = effective_bits_per_weight(N=2, M=4, bits=4, group_size=64, scale_bits=16.0)
# density = 0.5
# payload        = 0.5 * 4    = 2.0
# index overhead = 0.5 * ceil(log2(4)) = 0.5 * 2 = 1.0
# scale overhead = 16.0 / 64  = 0.25
# bpw = 2.0 + 1.0 + 0.25 = 3.25
```

## What the gate checks

The grader evaluates the exact formula above independently in NumPy
for several seeded `(N, M, bits, group_size, scale_bits)` configurations
(dense baselines with `N == M`, aggressive `1:4`/`2:8` patterns, and a
range of `group_size`/`scale_bits` combinations).

`rel_err` is the worst-case relative error between your returned `bpw`
and the oracle's, across all configurations (must be `<= 1e-9`) — this
is exact deterministic arithmetic (`M` is always a power of two in the
test configs, so `ceil(log2(M))` is exact), so any real formula
mismatch — forgetting a term, using `bits` instead of `density*bits`,
or amortizing the scale over kept rather than original weights —
produces an error far above tolerance.
