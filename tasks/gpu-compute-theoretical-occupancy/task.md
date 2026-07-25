## Context

A Streaming Multiprocessor (SM) can run several thread blocks
concurrently, but only as many as fit under *every* resource limit at
once:

- **Registers**: the SM has a fixed register file (`max_regs_per_sm`);
  each of a block's `threads_per_block` threads holds
  `regs_per_thread` registers for as long as it's resident.
- **Shared memory**: the SM has a fixed shared-memory budget
  (`max_shared_bytes_per_sm`); each resident block reserves
  `shared_bytes_per_block` of it.
- **Thread slots**: hardware caps how many threads can be resident at
  once (`max_threads_per_sm`).
- **Block slots**: hardware also caps how many *blocks* can be resident
  at once, independent of the above (`max_blocks_per_sm`).

The number of blocks that can actually be resident together is the
**minimum** of what each limit alone would allow:

$$
\text{active\_blocks} = \min\!\left(
  \left\lfloor \frac{\text{max\_regs\_per\_sm}}{\text{regs\_per\_thread} \cdot \text{threads\_per\_block}} \right\rfloor,\;
  \left\lfloor \frac{\text{max\_shared\_bytes\_per\_sm}}{\text{shared\_bytes\_per\_block}} \right\rfloor,\;
  \left\lfloor \frac{\text{max\_threads\_per\_sm}}{\text{threads\_per\_block}} \right\rfloor,\;
  \text{max\_blocks\_per\_sm}
\right)
$$

(when `shared_bytes_per_block` is `0`, that term is unconstrained --
treat it as `max_blocks_per_sm`). Each active block contributes
$\lceil \text{threads\_per\_block} / 32 \rceil$ warps, and **occupancy**
is the resulting active-warp count as a fraction of the SM's warp
capacity, `max_threads_per_sm / 32`.

A common mistake is to only account for registers, threads, and the
block cap and forget shared memory (or vice versa) -- but a kernel that
requests a lot of shared memory per block can be shared-memory-bound
even when it would have plenty of register and thread headroom left
over.

## Task

Implement:

```python
def compute_occupancy(
    regs_per_thread: int,
    shared_bytes_per_block: int,
    threads_per_block: int,
    max_regs_per_sm: int,
    max_shared_bytes_per_sm: int,
    max_threads_per_sm: int,
    max_blocks_per_sm: int,
) -> tuple[int, float]:
    ...
```

Return `(active_warps_per_sm, occupancy_fraction)`:

1. `active_blocks` = the four-way minimum above (integer division,
   floor, at each term).
2. `warps_per_block = ceil(threads_per_block / 32)`.
3. `active_warps_per_sm = active_blocks * warps_per_block`.
4. `occupancy_fraction = active_warps_per_sm / (max_threads_per_sm / 32)`.

## Example

`regs_per_thread=16, shared_bytes_per_block=16384, threads_per_block=128`
against `max_regs_per_sm=65536, max_shared_bytes_per_sm=49152,
max_threads_per_sm=2048, max_blocks_per_sm=32`:

- by registers: `65536 // (16*128) = 32` blocks.
- by shared memory: `49152 // 16384 = 3` blocks.
- by threads: `2048 // 128 = 16` blocks.
- `active_blocks = min(32, 3, 16, 32) = 3` -- shared memory is the
  binding limiter here, well below what registers or thread slots alone
  would allow.
- `warps_per_block = ceil(128/32) = 4`, so `active_warps_per_sm = 12`.
- `occupancy_fraction = 12 / (2048/32) = 12/64 = 0.1875`.

## What the gate checks

`check.py` runs `compute_occupancy` on 5 fixed configurations against
fixed SM limits (`max_regs_per_sm=65536`,
`max_shared_bytes_per_sm=49152`, `max_threads_per_sm=2048`,
`max_blocks_per_sm=32`) -- one where each of the four possible limiters
(registers, shared memory, thread count, the flat block cap) binds, plus
one tuned to land at exactly `100%` occupancy -- and compares the
returned `(active_warps_per_sm, occupancy_fraction)` tuple against the
same closed-form reference for each. `exact_match = 1.0` only if every
case matches exactly. A solution that omits the shared-memory term
(`min` over only registers, threads, and the block cap) matches on four
of the five cases but returns `(64, 1.0)` instead of the true `(12,
0.1875)` on the shared-memory-bound case.
