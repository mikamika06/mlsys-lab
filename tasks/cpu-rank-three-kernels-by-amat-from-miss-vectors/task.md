## Context

Average memory access time (AMAT) folds a whole cache hierarchy's hit
and miss rates into one number comparable across kernels:

$$\text{AMAT} = \text{L1\_HIT} + \text{L1\_miss\_rate} \times \big(\text{L2\_HIT} + \text{L2\_miss\_rate} \times \text{MEM\_PENALTY}\big)$$

where the L2 miss rate is *local* — the fraction of L1 **misses** (not
all accesses) that also miss L2. A kernel with more raw L1 misses is not
necessarily worse: if almost all of those misses are caught by L2 (low
local L2 miss rate), its AMAT can still beat a kernel with fewer L1
misses that mostly blow through to DRAM.

## Task

Implement

```cpp
void rank_by_amat(const long* accesses, const long* l1_misses, const long* l2_misses,
                   double* amat_out, int* rank_out);
```

For each of 3 kernels (`k` in `0..2`), with fixed costs `L1_HIT=1`,
`L2_HIT=10`, `MEM_PENALTY=100` cycles:

$$\text{L1\_miss\_rate} = \frac{\text{l1\_misses}[k]}{\text{accesses}[k]}, \qquad \text{L2\_miss\_rate} = \frac{\text{l2\_misses}[k]}{\text{l1\_misses}[k]}$$

Write `amat_out[k]` per the formula above, then write the 3 kernel ids
into `rank_out[0..3)`, sorted by ascending AMAT (fastest first).

## Example

Kernel 2 has the most L1 misses (`500` of `1000` accesses) but nearly all
of them hit L2 (`10/500 = 2%` local miss rate): `AMAT = 1 + 0.5*(10 +
0.02*100) = 7.0`. Kernel 0 has fewer L1 misses (`200`) but a worse local
L2 miss rate (`25%`): `AMAT = 1 + 0.2*(10 + 0.25*100) = 8.0` — worse than
kernel 2 despite fewer L1 misses. Kernel 1, with the fewest L1 misses
overall, comes out fastest at `AMAT = 5.5`. Ranked ascending:
`1, 2, 0`.

## What the gate checks

`exact_match`: the driver prints all 3 AMAT values and the full ranking
for one fixed set of miss vectors. Using the global (not local) L2 miss
rate, swapping which rate multiplies which penalty, or sorting by L1
miss count instead of AMAT all change the printed ranking; a starter
returning `0, 1, 2` in kernel-id order fails since the real order is
`1, 2, 0`.
