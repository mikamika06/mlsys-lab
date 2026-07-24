## Context

On NVIDIA GPUs a Streaming Multiprocessor (SM) executes warps
(groups of 32 threads) concurrently.  Hardware imposes four independent
upper bounds on how many warps can be resident at once:

1. **Register file.** Each thread consumes a fixed number of registers.
   The SM has `max_regs` registers total, so at most
   $B_{\text{regs}} = \lfloor \texttt{max\_regs} / (\texttt{block\_size}
   \times \texttt{regs\_per\_thread}) \rfloor$ blocks can fit from a
   register standpoint.

2. **Shared memory.** Each block may request `smem_per_block` bytes.
   With `max_smem` bytes available,
   $B_{\text{smem}} = \lfloor \texttt{max\_smem} /
   \texttt{smem\_per\_block} \rfloor$ (or unlimited when
   `smem_per_block == 0`).

3. **Block slot limit.** The SM can hold at most `max_blocks` blocks
   simultaneously.

4. **Warp slot limit.** The SM can hold at most `max_warps` warps
   in total.

A block of `block_size` threads occupies
$W_{\text{block}} = \lceil \texttt{block\_size} / 32 \rceil$ warps.
The actual number of concurrent blocks is the tightest of the three
block-level limits:

$$B_{\text{eff}} = \min\!\bigl(B_{\text{regs}},\; B_{\text{smem}},\; \texttt{max\_blocks}\bigr)$$

and the resulting active warp count is

$$W_{\text{active}} = \min\!\bigl(B_{\text{eff}} \times W_{\text{block}},\; \texttt{max\_warps}\bigr).$$

Occupancy — the ratio of active warps to the hardware maximum —
determines how well the SM can hide memory latency.  Higher occupancy
generally improves throughput up to the point where register pressure
forces spills to local memory.

## Task

Implement `max_active_warps`:

```python
def max_active_warps(
    regs_per_thread: int,
    smem_per_block: int,
    block_size: int,
    max_regs: int = 65536,
    max_smem: int = 98304,
    max_warps: int = 64,
    max_blocks: int = 48,
) -> int:
    ...
```

Return the maximum number of active warps per SM given the per-thread
register count, per-block shared-memory byte count, thread-block size,
and the four SM resource limits.  All inputs are non-negative integers.
Warp size is 32.

The default SM limits correspond to a typical modern architecture
(65 536 registers, 96 KiB shared memory, 64 warp slots, 48 block slots).

## Example

```python
# 128 threads, 32 regs each, no shared memory
max_active_warps(32, 0, 128)  # → 64
# Registers: ⌊65536/(128×32)⌋ = 16 blocks × 4 warps = 64  (hit warp cap)

# 256 threads, 64 regs each, no shared memory
max_active_warps(64, 0, 256)  # → 32
# Registers: ⌊65536/(256×64)⌋ = 4 blocks × 8 warps = 32

# Heavy shared memory
max_active_warps(16, 32768, 128)  # → 12
# Shared memory: ⌊98304/32768⌋ = 3 blocks × 4 warps = 12
```

## What the gate checks

Ten parameterised cases cover register-limited, smem-limited,
block-slot-limited, and warp-cap-limited scenarios, including
non-warp-aligned block sizes and custom SM limits.  Each case is
graded by `exact_match` against an independent reference that applies
the ceiling and floor formulas shown above; returning the wrong
integer for any case fails the gate.
