## Context

GEMM autotuners (CUTLASS, Triton, cuBLASLt) pick a CTA tile shape $(BM, BN)$ from a
short list of candidates before launching a matmul $C_{M\times N} = A_{M\times K} B_{K\times N}$.
Each tile shape partitions the output into a grid of

$$
\text{CTAs} = \left\lceil \frac{M}{BM} \right\rceil \cdot \left\lceil \frac{N}{BN} \right\rceil
$$

independent thread blocks, one per output tile. A GPU with `num_SMs` streaming
multiprocessors runs at most one CTA per SM at a time (assume single occupancy),
so the CTAs are scheduled in

$$
\text{waves} = \left\lceil \frac{\text{CTAs}}{\text{num\_SMs}} \right\rceil
$$

sequential waves. Because every wave costs roughly the same wall-clock time no
matter how full it is, a config whose last wave leaves most SMs idle is
**wave-quantized**: it pays for a whole extra wave to do a sliver of work. This
is the dominant effect real autotuners score first.

The second-order effect is **tile quantization**: tiles that don't evenly
divide $M$ and $N$ compute padded output past the true edge. The padded output
area is

$$
\text{tile\_area} = \left(\left\lceil \frac{M}{BM}\right\rceil BM\right)\left(\left\lceil \frac{N}{BN}\right\rceil BN\right),
\qquad
\text{waste} = \text{tile\_area} - M N \;\ge 0 .
$$

Combine the two into a single scalar cost, integer part = waves, fractional
part = the tile-waste fraction (always in $[0,1)$ since $\text{waste} < \text{tile\_area}$):

$$
\text{cost}(BM, BN) = \text{waves} + \frac{\text{waste}}{\text{tile\_area}} .
$$

Because the fractional part is strictly less than 1, comparing `cost` values
is equivalent to a lexicographic comparison on `(waves, waste_fraction)`:
configs are first ranked by number of waves, and only compared on padding
waste when their wave counts tie. `K` (the reduction dimension) scales the
useful and the wasted FLOPs of every candidate by the same factor, so it does
not change which tile wins — it is kept in the signature only because real
autotuners take it (e.g. for register/shared-memory feasibility checks that
are out of scope here).

## Task

Implement `select_autotune_tile`:

```python
def select_autotune_tile(M: int, N: int, K: int, num_SMs: int, candidates: list[tuple[int, int]]) -> tuple[int, list[float]]:
    ...
```

* `M, N, K` — GEMM problem shape ($C_{M\times N} = A_{M\times K}B_{K\times N}$).
* `num_SMs` — number of streaming multiprocessors on the target GPU.
* `candidates` — a sequence of `(BM, BN)` integer tile-shape tuples.

For every candidate compute `cost(BM, BN)` exactly as defined above, then
return `(best_idx, costs)` where:

* `costs` is a 1-D `float64` list of length `len(candidates)`, `costs[i]`
  holding the cost of `candidates[i]`;
* `best_idx` is the `int` index of the candidate with the smallest cost
(use `min` with `key`, which resolves exact ties by taking the first index —
  the same convention the oracle uses).

## Example

```python
M, N, K, num_SMs = 130, 130, 64, 4
candidates = [(64, 64), (128, 128)]

best_idx, costs = select_autotune_tile(M, N, K, num_SMs, candidates)
print(best_idx, costs)
# (64,64):  CTAs = ceil(130/64)*ceil(130/64) = 3*3 = 9, waves = ceil(9/4) = 3
#           tile_area = 192*192 = 36864, waste = 36864 - 16900 = 19964
#           cost = 3 + 19964/36864 ≈ 3.5416
# (128,128): CTAs = 2*2 = 4, waves = ceil(4/4) = 1
#           tile_area = 256*256 = 65536, waste = 65536 - 16900 = 48636
#           cost = 1 + 48636/65536 ≈ 1.7423
# -> best_idx = 1 (128,128 wins: far fewer waves, even though it wastes more area)
```

## What the gate checks

* **argmin_agreement** — across many randomly generated `(M, N, K, num_SMs,
  candidates)` problems, the fraction of cases where your `best_idx` matches
the Python oracle's `min` with `key` over the same cost formula. Must be `1.0`.
* **cost_rel_err** — the relative L2 error between your full `costs` vector
  and the oracle's, concatenated across all cases. Must be `<= 1e-6`.
