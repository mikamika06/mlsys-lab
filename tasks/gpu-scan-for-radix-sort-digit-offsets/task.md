## Context

One pass of radix sort groups elements by a single digit (a few bits,
or here, a value already in `[0, num_digits)`), preserving each
digit-group's *internal* order — that's what makes repeated passes over
successive digits eventually produce a fully sorted array. Doing that
grouping directly, in one sweep, needs to answer a question for every
element before writing anything: "how many slots come before mine?"

A **prefix sum (scan)** answers exactly that. First, histogram how many
elements have each digit value. Then, an **exclusive scan** over that
histogram turns per-digit *counts* into per-digit *starting offsets*:
digit `d`'s block of output begins right where digit `d-1`'s block
ends — `offsets[d] = offsets[d-1] + hist[d-1]`, `offsets[0] = 0`. From
there, every element with digit `d` lands at `offsets[d]` plus however
many digit-`d` elements were already placed before it — a running
per-digit counter, incremented as you scatter each one, walking the
input in its original order (that's the "stable" part: two elements
with the same digit keep their relative order, because whichever one
appeared first in the input claims the lower offset).

## Task

Implement, in `solve.cu`:

```cuda
__global__ void radix_scatter(const float* keys, const float* digits, float* out,
                               int n, int num_digits);
```

`digits[i]` is `keys[i]`'s digit for this pass, already extracted, an
integer in `[0, num_digits)`. Using the two provided
`__shared__ float hist[16]` / `offsets[16]` buffers (sized for up to 16
digit values):

1. Zero `hist[0..num_digits)`, then histogram: for each `i`, increment
   `hist[digits[i]]`.
2. Exclusive scan `hist` into `offsets`: `offsets[0] = 0`, and for
   `d >= 1`, `offsets[d] = offsets[d-1] + hist[d-1]`.
3. Zero `hist` again and reuse it as a per-digit cursor. Walk `i` from
   `0` to `n-1` **in order**: `pos = offsets[digits[i]] + hist[digits[i]]`,
   `out[pos] = keys[i]`, then increment `hist[digits[i]]`.

## Example

`keys = [10, 11, 12, 13]`, `digits = [1, 0, 1, 0]`, `num_digits = 2`.
Histogram: `hist = [2, 2]`. Offsets: `offsets = [0, 2]`. Scatter in
order: `keys[0]=10` (digit 1) goes to `offsets[1]+0 = 2`; `keys[1]=11`
(digit 0) goes to `offsets[0]+0 = 0`; `keys[2]=12` (digit 1) goes to
`offsets[1]+1 = 3`; `keys[3]=13` (digit 0) goes to `offsets[0]+1 = 1`.
Result: `out = [11, 13, 10, 12]` — digit-0 keys (`11, 13`) keep their
original relative order, and so do digit-1 keys (`10, 12`).

## What the gate checks

The grader builds a fixed 20-element fixture (`keys = 0..19`, random
digits in `[0, 4)`), launches `radix_scatter`, and compares the
scattered output against `sorted(keys, key=lambda i: digits[i])`
(Python's `sorted` is itself stable, so this is exactly the reference a
correct stable radix pass must match). It requires

$$
\mathrm{exact\_match} = 1 \iff \text{every one of the 20 output positions matches the stable-sorted reference exactly}
$$

Getting the histogram and scan right isn't enough on its own — the
scatter phase has to walk the input in its *original* order (not, say,
grouped by digit) for the placement to come out stable; reusing
`hist` as the running cursor without re-zeroing it after the scan step,
or scattering out of order, both produce a scatter that groups by digit
correctly but scrambles same-digit elements relative to each other.
