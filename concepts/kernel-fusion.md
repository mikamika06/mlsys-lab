---
title: "What is kernel fusion?"
description: "Kernel fusion explained, with a measured global-memory transactions and mem_insts table for a fused vs unfused elementwise chain you can reproduce without a GPU, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is kernel fusion?

Kernel fusion is combining several separate GPU kernels into one, so that every intermediate
value stays in a register instead of being written to and read back from global memory. A
three-stage elementwise chain run as three kernels costs three reads and three writes; fused
into a single kernel it costs exactly one read and one write, no matter how many stages sit
between them. Below is that gap counted exactly, on a simulator that needs no GPU.

## How it works

A kernel launch has no memory of the launch before it beyond whatever it wrote to global
memory. So a chain of `K` elementwise operations — add, then scale, then ReLU, then bias —
run as `K` separate kernels must round-trip every intermediate through DRAM: kernel `k` reads
what kernel `k-1` wrote, and writes what kernel `k+1` will read. Each of those tensors is the
same size as the input, so the traffic is `K` reads plus `K` writes, and it grows linearly with
the number of stages regardless of how little arithmetic each stage does.

That last part is the whole problem. An elementwise op like `add` or `ReLU` does one flop per
element it touches — the [arithmetic intensity](../tasks/gpu-compute-arithmetic-intensity-of-an-op/task.md)
is about as low as it gets, so the kernel is
[memory-bound](../tasks/gpu-classify-memory-bound-vs-compute-bound/task.md) from the start.
Every kernel boundary you add is one more full pass over memory for work that would otherwise
cost nothing beyond a register add. Fuse the chain into one kernel — read the inputs once, run
every stage on values held in registers, write the output once — and the memory traffic stops
scaling with the number of stages at all.

Fusion does not fix a bad access pattern; it removes a round trip that a good access pattern
would still have to pay. The fused kernel still needs
[coalesced](memory-coalescing.md) loads and stores inside itself, or the transaction count
inside that one kernel is exactly as bad as an unfused one. The two effects stack: an
unfused, strided chain is the worst of both worlds, and a fused, coalesced one is the best.
Compilers apply the same idea automatically — [`torch.compile`](torch-compile.md)'s Inductor
backend fuses a run of pointwise ops into one Triton kernel whenever no
[graph break](../tasks/rwc-count-dynamo-graph-breaks-in-a-traced-function/task.md) forces it to
stop, and the same reasoning is why LayerNorm is normally written with its residual add fused
in rather than as two kernels — see
[layer normalization](rmsnorm-vs-layernorm.md) for the op itself.

Fusion is not a mechanical rule, though. If a tensor feeds more than one downstream op, fusing
it into every consumer means recomputing it per consumer instead of materializing it once — a
real [fusion boundary decision](../tasks/gpu-decide-the-fusion-boundary-when-not-to-fuse/task.md).
Reductions complicate it further: a softmax's normalizing sum needs every element first, so
fusing past it needs an online running statistic, which is why some op pairs are genuinely
[fusion barriers](../tasks/rwc-tag-pointwise-fusible-ops-vs-fusion-barriers-in-an-op-chain/task.md)
and others fuse in one straight pass.

## Global-memory transactions and mem_insts against chain length

The table chains `K` elementwise increments over one warp of 32 elements two ways: as `K`
separate kernel launches, each reading and writing the whole array, and as one kernel that
loops `K` times over a register before writing. `transactions` counts 128-byte segments
touched; `mem_insts` counts individual global loads/stores issued — neither is a clock, so the
count is identical on every machine.

| chain length K | unfused transactions | fused transactions | unfused mem_insts | fused mem_insts | ratio |
|---|---|---|---|---|---|
| 1 | 2 | 2 | 64 | 64 | 1 |
| 2 | 4 | 2 | 128 | 64 | 2 |
| 4 | 8 | 2 | 256 | 64 | 4 |
| 8 | 16 | 2 | 512 | 64 | 8 |
| 16 | 32 | 2 | 1,024 | 64 | 16 |
| 32 | **64** | **2** | **2,048** | **64** | **32** |

Reproduce it — no NVIDIA hardware required, the two `.cu` kernels are executed by the software
GPU in [`src/mlsys/sim/`](../src/mlsys/sim/):

```bash
pip install mlsys-lab
python3 - <<'PY'
from mlsys.sim import CudaProgram, GPU
import numpy as np

SRC = """
__global__ void step(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1.0f;
}

__global__ void fused(float* out, const float* in, int n, int chain_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        for (int k = 0; k < chain_len; k++) {
            x = x + 1.0f;
        }
        out[i] = x;
    }
}
"""
N, BLOCK = 32, 32
prog = CudaProgram(SRC)

for K in (1, 2, 4, 8, 16, 32):
    gpu = GPU(N)
    gpu.gmem[0:N] = np.arange(N, dtype=np.float64)
    u_tx = u_mi = 0
    for _ in range(K):
        m = prog.launch(gpu, "step", 1, BLOCK, {"out": 0, "in": 0, "n": N})
        u_tx += m["transactions"]
        u_mi += m["mem_insts"]

    gpu2 = GPU(2 * N)
    gpu2.gmem[0:N] = np.arange(N, dtype=np.float64)
    mf = prog.launch(gpu2, "fused", 1, BLOCK, {"out": N, "in": 0, "n": N, "chain_len": K})

    print(f"K={K:>2}  unfused_transactions={u_tx:>3}  fused_transactions={mf['transactions']}  "
          f"unfused_mem_insts={u_mi:>5}  fused_mem_insts={mf['mem_insts']}  "
          f"ratio={u_tx // mf['transactions']}")
PY
```

Read the fused column first: it is 2 transactions and 64 memory instructions at every chain
length, because the loop that adds the chain's `K` steps runs entirely in the register `x` and
touches memory only for the one read of `in[i]` and the one write of `out[i]`. The unfused
column instead grows in lock-step with `K` — each extra kernel launch adds one more full read
and write of the array — so the ratio between them is exactly `K`, not an approximation of it.
That ratio is the entire argument for fusion stated as a number: fusing an 8-stage chain does
not save "some" memory traffic, it saves a measured 8x on both counters, for arithmetic that
was already nearly free.

## Practise it

```bash
mlsys grade gpu-fuse-elementwise-chain-add-mul-relu
```

[That task](../tasks/gpu-fuse-elementwise-chain-add-mul-relu/task.md) has you write one CUDA-C
kernel computing `max(0, (a+b)*c)` and gates it on `max_abs_err <= 1e-9` for correctness and
`transactions <= 40` for fusion. The failing move the gate is built to catch: writing the
`(a+b)*c` intermediate to `out[i]` and reading it back before the `max`, which passes
correctness and still fails the transaction budget, because that "fused" kernel just relocated
the round trip instead of removing it.

Related, in increasing scope: [count the round trips an affine-then-ReLU fusion actually saves](../tasks/gpu-count-global-round-trips-saved-by-fusion/task.md),
[compare DRAM traffic for a fused vs unfused elementwise chain in closed form](../tasks/rwc-compare-dram-traffic-fused-vs-unfused-elementwise-chain/task.md),
[fuse an arbitrary elementwise op chain into a single pass](../tasks/rwc-fuse-an-elementwise-op-chain-into-a-single-pass-kernel/task.md),
[fold bias and GELU into a matmul's epilogue](../tasks/gpu-fuse-bias-gelu-into-a-matmul-epilogue/task.md),
[fuse LayerNorm with its residual add](../tasks/gpu-fuse-layernorm-residual-add/task.md),
[write GELU/SiLU as a fused pointwise kernel from scratch](../tasks/gpu-fused-gelu-silu-pointwise-from-scratch/task.md),
and the same idea in [Triton, fusing add-scale-activation into one program](../tasks/gpu-triton-fused-elementwise-add-scale-activation/task.md).
When the decision is which pairs to fuse at all rather than how, there's
[classifying fusable op pairs](../tasks/sys-classify-fusable-op-pairs/task.md) and
[the speedup ceiling from fusing two ops](../tasks/sys-speedup-ceiling-from-fusing-two-ops/task.md).

## Common mistakes

- **"Fusing" by relocating the round trip instead of removing it.** Writing an intermediate to
  `out[i]` and reading it back in the same kernel is numerically identical to two kernels and
  costs the same transactions — the gate above exists specifically because this looks fused in
  the source.
- **Assuming fusion always pays off.** When an intermediate feeds several downstream
  consumers, fusing it into all of them means recomputing it once per consumer instead of
  materializing it once; past a small number of consumers, the recompute cost beats the
  memory-traffic saving.
- **Forgetting that fusion and coalescing are separate axes.** A fused kernel with a strided
  access pattern inside it still pays the transaction penalty from
  [memory coalescing](memory-coalescing.md) — fusion cuts the number of passes, coalescing
  cuts the cost of each pass, and a slow fused kernel usually has the second problem, not the
  first.
- **Treating a reduction like an elementwise op.** A sum, a softmax normalization, or a
  LayerNorm's variance all need every element before they can finish, so fusing across one
  needs an online statistic, not a straight register carry — the naive version either fuses
  incorrectly or cannot fuse at all.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which lists the
compilation-and-export area this term sits closest to as **adjacent only** — documentation and
reference code, nothing that grades a fusion decision:

- **[KernelBench](https://github.com/ScalingIntelligence/KernelBench)** — its 100 Level-2 tasks
  are specifically fused-operator kernels (conv+bias+activation and similar), scored on
  correctness plus wall-clock speedup against a PyTorch reference on a real GPU. Closest
  existing match on subject matter; the score depends on whatever GPU you run it on, and
  nothing separates "faster because fused" from "faster because someone hand-tuned tile sizes".
- **[Triton's own tutorials](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)**
  build a fused softmax kernel step by step and benchmark it against PyTorch's native op —
  excellent worked example, no starter/reference split and no gate beyond eyeballing the
  benchmark plot.
- **[GPU MODE lectures](https://github.com/gpu-mode/lectures)** cover fusion as part of the
  Triton and CUTLASS material with runnable notebooks; nothing checks whether a learner's own
  attempt actually fused anything.
- **PMPP** (reference 3 below) discusses fusion opportunities in its GEMM and convolution
  chapters; end-of-chapter, no autograder, community solution repos only.
- None of the above turn "did this kernel actually stop round-tripping through global memory"
  into a number the way `transactions` and `mem_insts` do here.

## References

1. Williams, S., Waterman, A., Patterson, D., *Roofline: An Insightful Visual Performance
   Model for Multicore Architectures*, Communications of the ACM, 2009 — the arithmetic-
   intensity argument for why memory-bound elementwise chains are what fusion targets.
   https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S12/handouts/papers/RooflineVyNoYellow.pdf
2. Triton, *Fused Softmax* tutorial — a concrete worked fusion, max/sub/exp/sum/div collapsed
   into one kernel. https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
3. Kirk, D., Hwu, W., Hajj, I., *Programming Massively Parallel Processors*, 4th ed. — kernel
   fusion and memory-traffic reduction, covered across the GEMM and convolution chapters.
   https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors
