## Context

A tensor-core MMA instruction doesn't read its operands from memory the
way an ordinary load does — it expects each operand matrix already split
across the 32 lanes of a warp, a handful of elements per lane, in a fixed
layout the hardware defines. Before you can even issue the instruction,
every lane has to load *its own* pieces of `A` and `B` into local
registers ("fragments") in exactly the right positions. Get the packing
wrong and the MMA still runs — it just multiplies the wrong numbers
together, silently.

This task models that packing step for an `m16n8k16`-shaped operation: `A`
is 16x16, `B` is 16x8, and a 32-lane warp splits each of them by index,
grouped and offset the way real tensor-core ISAs distribute operands
across a warp (not claimed to match any one specific hardware revision
bit-for-bit — real MMA execution itself is outside what this software GPU
models):

$$
\text{groupID} = \left\lfloor \text{lane} / 4 \right\rfloor \ (0..7)
\qquad
\text{tid\_in\_group} = \text{lane} \bmod 4 \ (0..3)
$$

**A** (each lane holds 8 elements, index `k` in `[0, 8)`):
$$
\text{half} = \lfloor k/4 \rfloor, \quad \text{sub} = k \bmod 4
$$
$$
\text{row} = \text{groupID} + \text{half} \times 8, \qquad
\text{col} = \text{tid\_in\_group} \times 4 + \text{sub}
$$

**B** (each lane holds 4 elements, index `k` in `[0, 4)`):
$$
\text{row} = \text{tid\_in\_group} \times 4 + k, \qquad
\text{col} = \text{groupID}
$$

Every `(row, col)` of `A` is covered by exactly one `(lane, k)` — 8 groups
$\times$ 2 halves $\times$ 4 lanes $\times$ 4 subs $= 256 = 16\times16$ —
and every `(row, col)` of `B` likewise ($8 \times 4 \times 4 = 128 =
16\times8$).

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void pack_mma_fragment(float* fragA_out, float* fragB_out,
                                   const float* A, const float* B);
```

One block, 32 threads (one warp). For each lane `threadIdx.x`, write its 8
`A` elements to `fragA_out[lane*8 .. lane*8+7]` and its 4 `B` elements to
`fragB_out[lane*4 .. lane*4+3]`, using the index formulas above (`A` is
row-major `16x16`, so `A[row][col] = A[row*16 + col]`; `B` is row-major
`16x8`, so `B[row][col] = B[row*8 + col]`).

## Example

Lane 5: `groupID = 5/4 = 1`, `tid_in_group = 5%4 = 1`. Its `A` elements
(`k=0..7`): `half=0` for `k=0..3` gives `row=1`, `col=4+sub` for
`sub=0..3` → `A[1][4..7]`; `half=1` for `k=4..7` gives `row=9`,
`col=4+sub` → `A[9][4..7]`. Its `B` elements (`k=0..3`):
`row=4+k` → `B[4..7][1]` (column 1, the row it belongs to as
`groupID=1`).

## What the gate checks

`check.py` builds random `16x16` and `16x8` matrices, parses `solve.cu`,
and runs `pack_mma_fragment` on the software GPU (`arena.cuda_sim.GPU`)
with a 1-block, 32-thread launch. It requires `max_abs_err == 0.0` against
fragments built independently in Python from the same formula. Swapping
`groupID` and `tid_in_group` (i.e. `groupID = lane % 4`, `tid_in_group =
lane / 4`) still produces *some* permutation of `A`'s and `B`'s elements
across the 32 lanes — every element still gets read exactly once — but
almost every lane ends up holding a different set of elements than it
should, and the mismatch is caught immediately by the exact per-lane
comparison.
