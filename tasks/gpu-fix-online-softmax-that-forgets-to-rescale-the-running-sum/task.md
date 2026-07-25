## Context

A numerically-safe softmax subtracts the row's max before exponentiating
(`exp(x - max)` instead of `exp(x)`, so nothing overflows). An **online**
softmax computes that max *while streaming through the row*, without a
separate first pass to find it — which means the max can keep changing
partway through, and every partial sum accumulated so far was computed
relative to whatever the max *used to be*.

The fix is a rescale: whenever a new element pushes the running max `m`
up to `new_m`, the running sum `l` — every term of which is
`exp(x_i - m)`, on the OLD scale — has to be multiplied by
`exp(m - new_m)` *before* adding the new term's contribution
`exp(x - new_m)`. Skip that rescale and `l` silently mixes terms
computed on two different scales, so the final `exp(x - m) / l`
normalization comes out wrong for any row where the max isn't the very
first element.

## Task

Fix the online softmax kernel:

```cuda
__global__ void online_softmax(float* out, const float* in, int B, int N);
```

One thread per row (`row = blockIdx.x*blockDim.x + threadIdx.x`, guarded
by `row < B`). First pass, streaming `j = 0..N-1`: track `m` (running
max, start at a very negative number) and `l` (running sum). At each
step, `new_m = fmaxf(m, in[row*N+j])`, then

$$
l \leftarrow l \cdot e^{m - \text{new\_m}} + e^{x - \text{new\_m}}, \qquad m \leftarrow \text{new\_m}
$$

Second pass, streaming `j = 0..N-1` again: `out[row*N+j] = exp(x - m) / l`
using the FINAL `m` and `l` from the first pass.

## Example

Row `[1, 5, 2]`: step `j=0`: `m=-inf -> 1`, `l = 0*e^{-inf} + e^{0} = 1`.
Step `j=1`: `new_m = 5`; correctly, `l = 1 * e^{1-5} + e^{5-5} = e^{-4} + 1
\approx 1.0183`; `m=5`. Step `j=2`: `new_m = 5` (unchanged, no rescale
needed), `l \mathrel{+}= e^{2-5} \approx 1.0682`. Final:
`out = [e^{1-5}/l, e^{5-5}/l, e^{2-5}/l]`. Skip the rescale at `j=1` and
`l` stays `1 + e^{5-5} = 2` instead of `\approx 1.0183` — every output in
the row comes out low by roughly that same wrong factor.

## What the gate checks

`check.py` seeds a fixed random `8 x 32` input (values spread widely
enough that the running max updates several times per row, not just at
`j=0`), parses `solve.cu`, and launches `online_softmax` as `8` threads
(one block, one thread per row). It compares the output against a numpy
batch-softmax oracle (`exp(x - x.max(axis=1)) / sum(...)`, computed from
the same seeded input) and requires

$$
\mathrm{max\_abs\_err} = \max |{\text{out} - \text{oracle}}| \le 10^{-6}
$$

The unrescaled version (`l += exp(x - new_m)`, missing the
`l *= exp(m - new_m)` term) measures `max_abs_err \approx 0.25` — every
row whose max isn't at index `0` comes out on the wrong scale.
