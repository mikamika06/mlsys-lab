## Context

The roofline model caps a kernel's achievable throughput at whichever
resource it exhausts first:

$$
\text{attainable} = \min(\text{peak\_flops}, \text{AI} \times \text{peak\_bandwidth}), \qquad
\text{AI} = \frac{\text{flops}}{\text{bytes}}
$$

Two kernels with wildly different absolute FLOP counts can have the same
attainable throughput if their AI puts them both above the ridge point
(both simply get capped at `peak_flops`); two kernels with the same AI but
different absolute sizes always get the *same* attainable throughput
(size cancels out of the ratio). Ranking kernels by *raw FLOP count* or by
*raw AI* alone gives the wrong order the moment some kernels are
compute-bound (capped) and others aren't.

## Task

Implement

```cpp
void rank_kernels_by_attainable_perf(const double* flops, const double* bytes, int n,
                                      double peak_flops_per_sec, double peak_bytes_per_sec,
                                      int* rank_out);
```

For every kernel `i`, compute `ai = flops[i] / bytes[i]` and
`attainable = min(peak_flops_per_sec, ai * peak_bytes_per_sec)`. Write the
kernels' original indices into `rank_out[0..n)`, sorted by `attainable`
**descending** (highest attainable throughput first).

## Example

`peak_flops_per_sec=200, peak_bytes_per_sec=50` (ridge = 4 FLOP/byte).
Kernel A: `flops=2.0, bytes=0.1` → `ai=20`, `attainable=min(200,
20*50)=200` (compute-bound, capped). Kernel B: `flops=1.2, bytes=0.4` →
`ai=3`, `attainable=min(200, 3*50)=150` (memory-bound, below the ridge,
not capped). A ranks above B even though B's raw FLOP count is lower and
its AI is also lower — what matters is the capped attainable value, `200`
vs `150`.

## What the gate checks

`main.cpp` runs two fixed 5-kernel scenarios (different peak
FLOPs/bandwidth pairs, so different ridge points — no two kernels tie on
attainable throughput in either) and prints the resulting rank
permutation for each. The candidate's full stdout is compared
byte-for-byte (`exact_match = 1.0`) against the reference's. Sorting by
raw AI instead of the capped attainable value, or forgetting the `min()`
cap entirely (ranking purely by `ai * peak_bandwidth`), reorders any
kernel pair where one is compute-bound and the other memory-bound —
which is most of both fixtures.
