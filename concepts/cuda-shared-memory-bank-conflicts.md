---
title: "What are shared memory bank conflicts?"
description: "Shared memory bank conflicts explained, with a measured smem_waves-vs-stride table you can reproduce without a GPU, plus a graded padding exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What are shared memory bank conflicts?

Shared memory bank conflicts happen when two or more lanes of a warp request different
addresses that hash to the same one of the 32 on-chip banks, so the hardware serializes what
should be a one-cycle access into as many cycles as the worst-hit bank has distinct addresses.
Below, an unpadded 32-word tile column read costs 33 shared-memory waves for one instruction;
one word of padding drops it to 2. That cliff, and the reason it does not stay fixed as padding
grows, is measured next on the software GPU used throughout this bank.

## How it works

Shared memory is on-chip and word-interleaved across 32 banks: word index `w` lives in bank
`w % 32`, and each bank can service exactly one address per cycle. A warp's shared-memory
instruction is free — one wave — only if its 32 lanes touch 32 different banks, or all agree
on the exact same address, which the hardware serves as a single **broadcast** rather than a
conflict. Anything in between serializes: if `n` lanes request `n` different words that all
land in one bank, that instruction takes `n` cycles instead of one, and the [conflict-degree
task](../tasks/gpu-compute-shared-memory-bank-conflict-degree/task.md) makes exactly that
broadcast-vs-conflict distinction the thing being graded.

This is the on-chip sibling of [memory coalescing](memory-coalescing.md): both are "did the
32 addresses a warp just issued spread out the way the hardware wants," but coalescing counts
128-byte DRAM segments touched by a *global* load, while a bank conflict counts 4-byte words
colliding inside *shared* memory — a narrower unit, and one that a programmer controls
directly, because the tile's layout in shared memory is something the kernel chooses. It is
also worth separating from [false sharing](false-sharing.md), which is the same "granularity
smaller than the access" idea on a CPU's coherence protocol: there, colliding inside a line
costs a cache invalidation between cores; here, colliding inside a bank costs a serialized
cycle inside one warp, and there is no correctness risk either way, only wasted cycles.

The pattern that reliably produces conflicts is a **tile read down its own stride**: stage a
`row x row_stride` tile in shared memory, then have each lane of a warp read a different row
at a fixed column. If `row_stride` is a multiple of 32 — the natural choice for a 32-wide
tile — every lane's address is a multiple of 32 apart, so all 32 land in bank 0. This shows up
in exactly the same two places coalescing does: a [transpose staged through shared
memory](../tasks/gpu-1-padding-to-remove-transpose-bank-conflicts/task.md), and a [GEMM tile
staged the same way](../tasks/gpu-conflict-free-shared-staging-for-a-gemm-tile/task.md), where
the reduction loop reads a column of the staged operand. The fix in both cases is the same
one-line change: pad the row stride so it is no longer a multiple of 32.

## Bank-conflict waves measured against row stride

The kernel below stages a 32-word tile with one conflict-free write per lane, then has the
warp read it back down a column at a stride that varies from 32 upward, counting the
`smem_waves` and `smem_insts` the simulator reports for that launch.

| row stride | smem_waves | smem_insts |
|---|---|---|
| 32 | **33** | 64 |
| 33 | **2** | 64 |
| 34 | 3 | 64 |
| 35 | 2 | 64 |
| 36 | 5 | 64 |
| 40 | 9 | 64 |
| 48 | 17 | 64 |
| 64 | **33** | 64 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
from mlsys.sim import CudaProgram, GPU
import numpy as np

SRC = """
__global__ void col_read(float* out, const float* seed, int stride) {
    __shared__ float tile[2048];
    int t = threadIdx.x;
    tile[t] = seed[t];          // conflict-free fill: one word per bank
    __syncthreads();
    out[t] = tile[t * stride];  // column read: bank (t*stride) % 32
}
"""
N = 32
for stride in (32, 33, 34, 35, 36, 40, 48, 64):
    gpu = GPU(2 * N, smem_size=2048)
    gpu.gmem[0:N] = np.arange(N, dtype=np.float64)
    m = CudaProgram(SRC).launch(gpu, "col_read", 1, N,
                                 {"out": N, "seed": 0, "stride": stride})
    print(f"stride={stride:>3}  smem_waves={m['smem_waves']:>3}  "
          f"smem_insts={m['smem_insts']:>4}")
PY
```

Read `smem_waves` as **one fill wave plus the read's conflict degree**: the fill is always
conflict-free (32 lanes, 32 distinct banks, one wave), so subtract 1 to get the read alone —
32-way at stride 32, a genuine cliff to 1-way at stride 33, back up to a 2-way conflict at
stride 34, and so on following `gcd(stride, 32)`. `smem_insts` is flat at 64 the entire time,
which is the point: padding changes nothing about how many shared-memory instructions the
kernel issues, only how many cycles each one costs. And the table does not stay fixed once
you start padding — stride 64 is a multiple of 32 again, so doubling the padding all the way
round the bank count reintroduces the full 32-way conflict the padding was meant to remove.

## Practise it

```bash
mlsys grade gpu-1-padding-to-remove-transpose-bank-conflicts
```

[That task](../tasks/gpu-1-padding-to-remove-transpose-bank-conflicts/task.md) gates a real
`.cu` transpose on `max_abs_err <= 1e-09` for correctness and `smem_wave_ratio <= 1.05` for
the access pattern, where the denominator is a conflict-free wave count the grader measures
itself by running its own padded kernel through the same simulator, never a hardcoded
constant. The shipped starter declares its shared tile as `tile[1024]` with `row * 32 + col`
indexing on both the store and the transposed load — numerically correct, comment says so
outright, and it still fails, because every column read after the barrier lands all 32 lanes
of a warp in one bank. Changing only the stride to `n + 1` (`tile[1056]`, `row * 33 + col`)
is the entire fix; the transpose logic itself does not change at all.

In increasing difficulty:
[predict conflict degree from a stride](../tasks/sys-predict-padding-effect-on-bank-conflicts/task.md)
(no code, just the `gcd`-with-32 arithmetic),
[compute bank-conflict degree for arbitrary warps](../tasks/sys-shared-memory-bank-conflict-count/task.md),
[probe broadcast vs conflict on a fixed access pattern](../tasks/gpu-compute-shared-memory-bank-conflict-degree/task.md),
[pad a second, differently-shaped transpose tile](../tasks/gpu-1-padding-for-a-conflict-free-transpose-tile/task.md),
and [stage a GEMM tile conflict-free in both operands](../tasks/gpu-conflict-free-shared-staging-for-a-gemm-tile/task.md).

## Common mistakes

- **Padding by a multiple of 32.** Stride 64 measures the same 33 waves as stride 32 in the
  table above — padding has to change the stride's residue mod 32, and adding exactly one
  bank-width of padding changes nothing.
- **Assuming `+1` is a universal fix.** It works because 33 is odd and 32 is a power of two,
  so `gcd(33, 32) = 1`. If the unpadded stride is already odd, adding 1 makes it even and can
  reintroduce a conflict — the padding amount has to be chosen to leave the stride coprime
  with 32, not applied by reflex.
- **Confusing a broadcast with a conflict.** Multiple lanes reading the *exact same* address
  cost one wave, not a conflict; the conflict-degree task above exists specifically because
  this distinction is easy to get backwards when counting by hand.
- **Reading `smem_insts` as the cost.** It stayed at 64 across every row in the table above.
  Padding never reduces how many shared-memory instructions a kernel issues — it only changes
  how many cycles each one takes, which is what `smem_waves` counts and `smem_insts` does not.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[GPU-Puzzles](https://github.com/srush/GPU-Puzzles)** — 14 notebook puzzles that build
  the same shared-memory intuition through NUMBA's CUDA JIT on a real GPU runtime, but its
  self-check is a NumPy-equality assertion; it never counts a bank conflict.
- **[LeetGPU](https://leetgpu.com/)** and **[Tensara](https://tensara.org/)** — real-hardware
  judges with transpose and GEMM among their problem sets, scored on correctness plus
  wall-clock or relative speed against a reference. An unpadded and a padded kernel both pass
  if both are fast enough on the GPU you happened to draw; neither reports why one is faster.
- **[SW Online Judge](https://swforces.com/)** — genuinely no GPU needed, via a CUDA-to-OpenMP
  transpiler, but its own docs state it verifies correctness only, so a 32-way-conflicted
  kernel and a padded one look identical to it.
- **[LeetCUDA](https://github.com/xlite-dev/LeetCUDA)** — 200+ reference kernels including
  tiled GEMM and transpose variants that already pad correctly; the best place to see the
  fix applied at production quality once you understand why it is there.
- **Programming Massively Parallel Processors** (reference 2) is the textbook this whole
  topic traces back to, and its end-of-chapter exercises on tiling are the closest print
  analogue — worked by hand, never auto-graded.

## References

1. NVIDIA, *CUDA C++ Programming Guide*, §"Shared Memory" — bank layout and the broadcast
   exception. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
2. Kirk, D. B. & Hwu, W. W., *Programming Massively Parallel Processors*, 4th ed., ch. 5,
   "Memory Architecture and Data Locality" — tiling and shared-memory bank conflicts.
   https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors
3. NVIDIA Developer Blog, *Using Shared Memory in CUDA C/C++*.
   https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
