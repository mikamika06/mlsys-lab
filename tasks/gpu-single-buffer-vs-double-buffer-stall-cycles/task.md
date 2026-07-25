## Context

Processing `T` tiles, each needing `load_cycles` to fetch from global
memory and `compute_cycles` to process once it's in hand, can be
scheduled two ways. With a **single buffer**, there's nowhere to put
the next tile's data until the current tile is done being computed on
— every load and every compute happen strictly one after another, no
exceptions. With a **double buffer**, one buffer can be loading the
*next* tile while the *other* buffer's contents are being computed on
right now — the load and the compute for adjacent tiles overlap.

Overlap doesn't make either operation faster individually — it makes
the slower of the two "absorb" the faster one for free. Once tile 0 is
loaded (an unavoidable first wait, since there's nothing yet to overlap
it with) and until the very last tile's compute (nothing left to
overlap it with either), every step in between costs whichever of
`load_cycles` / `compute_cycles` is larger, not their sum.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void buffering_cycles(int T, int load_cycles, int compute_cycles, float* out);
```

Write into `out[0]` the single-buffered total:
`T * (load_cycles + compute_cycles)`. Write into `out[1]` the
double-buffered total:
`load_cycles + (T-1) * max(load_cycles, compute_cycles) + compute_cycles`.

## Example

`T=10, load_cycles=50, compute_cycles=30`: single-buffered:
`10 * (50+30) = 800`. Double-buffered: prologue `50`, then `9` steady
steps at `max(50,30)=50` each (`450`), then epilogue `30` —
`50 + 450 + 30 = 530`. Double buffering here is bound by the *loads*
(the slower operation), not free, but still cuts total time by a third.

## What the gate checks

The grader launches `buffering_cycles` for 5 fixed `(T, load_cycles,
compute_cycles)` scenarios and compares both outputs against an
independently computed oracle. It requires

$$
\mathrm{exact\_match} = 1 \iff \text{both outputs match the oracle on every one of the 5 scenarios}
$$

Across all 5, the double-buffered total is always lower — never by
more than the single-buffered version's own smaller half (`load_cycles`
or `compute_cycles`, whichever is smaller), since overlap can hide the
faster operation completely but can never make the slower one
disappear.
