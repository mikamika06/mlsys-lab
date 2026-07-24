## Context

A classic way to *measure* a CPU's real cache hierarchy is a working-set
sweep: repeatedly touch an array, doubling its size each time, and record
the average per-access latency. As long as the array fits in L1, latency
stays near L1's cost; once it outgrows L1 but still fits L2, latency jumps
up and plateaus at L2's cost; and so on through L3 and finally DRAM. Each
transition between plateaus is a **knee** — and its position tells you
exactly how big that cache level is.

Real measurements are noisy, so a knee isn't "any change at all" — it's a
**relative jump** large enough that it can only be a level transition, not
ordinary jitter within a plateau.

## Task

Implement

```cpp
int detect_knees(const double* latency, int n, double rel_threshold, int* out_indices);
```

which scans a latency array of `n` samples (ordered by increasing
working-set size) and, for every `i` in `[1, n)`, computes

$$
\text{rel}(i) = \frac{\text{latency}[i] - \text{latency}[i-1]}{\text{latency}[i-1]}
$$

If $\text{rel}(i) > \text{rel\_threshold}$, `i` is a knee: append it to
`out_indices` (a caller-provided buffer with room for at least `n-1`
ints), in increasing order. Return the total number of knees appended.

## Example

`latency = [4.02, 4.06, 4.00, 4.04, 4.08, 4.02, 12.08, 12.00, 12.04, ...]`,
`rel_threshold = 0.5`: between index 5 (`4.02`) and index 6 (`12.08`),
$\text{rel} = (12.08 - 4.02)/4.02 \approx 2.0 > 0.5$ — index `6` is a
knee. Between index 6 (`12.08`) and index 7 (`12.00`),
$\text{rel} \approx -0.007$ — nowhere close to the threshold, not a knee,
even though the sign is negative (a small dip is not a jump).

## What the gate checks

`main.cpp` builds two fixed latency-ladder fixtures deterministically (no
`rand()`): a 20-sample sweep from 1 KiB to 512 MiB modeling L1/L2/L3/DRAM
(3 true knees), and a 12-sample sweep from 1 KiB to 2048 KiB modeling
L1/L2/DRAM (2 true knees) — each with a per-sample jitter of a constant
+-3% of that sample's OWN base latency (small and relative, exactly like
real measurement noise), so plateaus aren't perfectly flat — and prints
the knee count and indices found at each fixture's threshold. The
candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`)
against the reference's. Because the jitter scales with the base latency,
its *relative* size stays a constant ~3-6% everywhere (safely under every
`rel_threshold` used), but its *raw* (cycle) size is tiny at L1 and large
at DRAM: comparing a raw difference against `rel_threshold` instead of a
relative one reports several spurious knees inside the high-latency
plateaus (8 and 4 "knees" instead of the true 3 and 2) while still
catching the real transitions — passing nowhere near the reference.
