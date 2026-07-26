---
title: "What is memory coalescing?"
description: "Memory coalescing explained, with a measured stride-vs-transactions table you can reproduce on any machine without a GPU, plus a graded CUDA exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is memory coalescing?

Memory coalescing — the hardware also calls the good case a **coalesced memory access** — is
the merging of the 32 separate global-memory addresses requested by a warp into as few
128-byte transactions as possible. When the addresses are
consecutive, one transaction serves all 32 threads; when they are strided by 32 floats, it
takes 33 — the same useful bytes, sixteen times the read traffic. Below is that count
measured exactly, by a simulator that needs no GPU.

## How it works

A warp is 32 threads issuing one instruction together. When that instruction is a global
load, the memory subsystem does not fetch 32 individual values — it fetches whole aligned
128-byte segments, because DRAM is organised in bursts and a burst is the smallest thing
worth paying for. 128 bytes is exactly 32 four-byte floats, which is not a coincidence: the
segment size was chosen so that one perfectly aligned, consecutive warp read costs one
segment.

So the cost of a warp's load is not "32 reads". It is **the number of distinct 128-byte
segments the 32 addresses fall into.** If thread `t` reads `in[t]`, all 32 addresses land in
one segment and the load is *coalesced*. If thread `t` reads `in[t * 4]`, the addresses span
four segments — the same useful data arrives, and three quarters of the fetched bytes are
discarded.

This is the cache-locality problem with the politeness removed. On a CPU a cache-line fetch
is automatic and a strided loop merely runs slower; the analogous failure there is
[false sharing](false-sharing.md), where distinct addresses collide inside one line. On a GPU
the same access pattern is a visible, countable multiplier on memory traffic, and because
kernels are usually memory-bound rather than compute-bound, it is usually *the* thing that
decides how fast the kernel runs.

Two neighbouring effects are separate and are counted separately. Coalescing governs
**global** memory; the equivalent question for on-chip memory is shared-memory bank
conflicts, where the unit is one of 32 banks rather than a 128-byte segment — that is
[padding to remove transpose bank conflicts](../tasks/gpu-1-padding-to-remove-transpose-bank-conflicts/task.md).
And a warp whose threads take different branches suffers warp divergence, which multiplies
instruction issues rather than memory transactions. A slow kernel usually has more than one
of the three.

The standard fix, when the natural access pattern is strided — a column-major gather, a
transpose — is to stage the awkward access through shared memory: read global memory
coalesced, write it to a shared tile, then read the tile in whatever order the arithmetic
wants. That converts a global-transaction problem into a bank-conflict problem, which is
cheaper, and which padding the tile can then remove.

## Transactions measured against stride

The table varies only the stride of a global read in a one-warp kernel and counts the
128-byte transactions the access generates. Nothing here is timed: `transactions` counts
segments touched, so the numbers are identical on every machine.

| stride | transactions | cycles | divergences |
|---|---|---|---|
| 1 | **2** | 404 | 0 |
| 2 | 3 | 604 | 0 |
| 4 | 5 | 1,004 | 0 |
| 8 | 9 | 1,804 | 0 |
| 16 | 17 | 3,404 | 0 |
| 32 | **33** | 6,604 | 0 |

Reproduce it — no NVIDIA hardware required, the `.cu` is executed by the software GPU in
[`src/mlsys/sim/`](../src/mlsys/sim/):

```bash
pip install mlsys-lab
python3 - <<'PY'
from mlsys.sim import CudaProgram, GPU
import numpy as np

SRC = """
__global__ void strided(double* out, const double* in, int n, int stride) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int j = t * stride;
    if (j < n) out[t] = in[j] * 2.0;
}
"""
N, BLOCK = 1024, 32
for stride in (1, 2, 4, 8, 16, 32):
    gpu = GPU(2 * N)
    gpu.gmem[0:N] = np.arange(N, dtype=np.float64)
    m = CudaProgram(SRC).launch(gpu, "strided", 1, BLOCK,
                                {"out": N, "in": 0, "n": N, "stride": stride})
    print(f"stride={stride:>3}  transactions={m['transactions']:>3}  cycles={m['cycles']}")
PY
```

Read the table as **one write transaction plus one read transaction per stride step**. At
stride 1 the warp's 32 reads fall inside a single segment, so the kernel costs 2 transactions
in total: one for the read, one for the coalesced write of `out[t]`. Each doubling of the
stride doubles the read cost while the write stays free, so by stride 32 the kernel pays 33
transactions to move the same 32 useful values. The `cycles` column rises with it because the
model charges each transaction the same memory latency; that linearity is a property of the
model, and on real hardware the curve bends once L2 and TLB effects appear.

The most useful consequence: **stride 2 already costs 50% more transactions than stride 1.**
Coalescing is not a cliff you fall off at some large stride — it degrades from the very first
step, which is why an `x[2*i]` indexing habit carried over from CPU code is expensive here
immediately.

## Practise it

```bash
mlsys grade gpu-ex-cuda-coalesced-scale
```

[That task](../tasks/gpu-ex-cuda-coalesced-scale/task.md) gates a real `.cu` you write on
two numbers: `max_abs_err <= 1e-09` for correctness and `transactions <= 20` for the access
pattern. The shipped starter computes the index and writes nothing, so it fails correctness
first. The second gate is the interesting one — a version that is numerically perfect but
indexes `in[i * stride]`, or that permutes threads to elements, passes `max_abs_err` and
still fails, because the warp's addresses no longer fall inside one segment.
**Correct-but-scattered is a failing answer here.**

In increasing difficulty:
[rank the access patterns](../tasks/gpu-rank-access-patterns-coalesced-scattered/task.md) (no code),
[coalesced vs strided](../tasks/gpu-coalesced-vs-strided-access/task.md) (count it yourself),
[fix a transpose that writes uncoalesced](../tasks/gpu-fix-a-transpose-that-writes-uncoalesced/task.md),
[relayout a column-major gather](../tasks/gpu-relayout-column-major-gather-to-coalesced-row-major/task.md),
and [a tiled transpose coalesced both ways](../tasks/gpu-shared-memory-tiled-transpose-coalesced-r-w/task.md).

## Common mistakes

- **Treating alignment as optional.** A consecutive read that starts at element 8 instead of
  element 0 straddles two segments and costs twice what it should — a 100% overhead on a
  pattern that looks perfectly coalesced in the source.
- **Coalescing the read and forgetting the write.** A transpose kernel typically reads
  `in[y*W + x]` coalesced and writes `out[x*H + y]` scattered. The write costs exactly as much
  as a read would.
- **Assuming struct-of-arrays is always the answer.** Switching AoS to SoA fixes coalescing
  only if the kernel does not use every field. If it touches one field of eight, the AoS
  layout gives you a stride of eight and SoA is the fix; if it touches all eight, the
  transaction count is the same and you have only made the code harder to read.
- **Reading `transactions` as a time.** It counts segments touched. It predicts bandwidth
  pressure, not wall-clock, and it deliberately ignores latency hiding — which is why it is
  reproducible and why the gates use it.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[GPU-Puzzles](https://github.com/srush/GPU-Puzzles)** — 14 notebook puzzles, the
  best-known resource in this space. Builds the same thread/block/shared-memory intuition;
  self-checks against a NumPy reference, and does not measure coalescing at all.
- **[LeetGPU](https://leetgpu.com/)** and **[Tensara](https://tensara.org/)** — browser
  judges running on real GPUs. Same problem topics (reduction, scan, transpose, GEMM), scored
  on correctness plus relative speed, so the number depends on shared hardware you do not
  control.
- **[SW Online Judge](https://swforces.com/)** — the closest cousin to this page's approach:
  real CUDA-C, genuinely no GPU needed, via a CUDA-to-OpenMP transpiler. Its own docs say
  "performance benchmarking is not available — the platform is for correctness verification
  only", so it catches a wrong answer and not a wrong access pattern.
- **[LeetCUDA](https://github.com/xlite-dev/LeetCUDA)** — 200+ reference kernels with
  benchmark tables. Nothing to submit; the best place to see what a genuinely fast kernel
  looks like once puzzles stop being useful.
- **NVIDIA's own guide**, reference 1 below, remains the primary source. Read it after
  measuring, not instead.

## References

1. NVIDIA, *CUDA C++ Best Practices Guide*, §9.2.1 "Coalesced Access to Global Memory".
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
2. NVIDIA Developer Blog, *How to Access Global Memory Efficiently in CUDA C/C++ Kernels*.
   https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
3. Drepper, U., *What Every Programmer Should Know About Memory*, 2007 — the DRAM burst
   mechanism, CPU side. https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
