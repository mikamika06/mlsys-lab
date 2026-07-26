---
title: "What is warp divergence?"
description: "Warp divergence explained, with a measured divergences-and-cycles table you can reproduce on any machine without a GPU, plus a graded CUDA exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is warp divergence?

Warp divergence is the penalty a GPU pays when the 32 lockstep threads of one
warp disagree about which side of a conditional to take, forcing the hardware
to serialise every distinct path instead of running one shared instruction
stream. Below, a single dissenting lane out of 32 turns out to cost exactly
as much as fifteen — 601 modelled cycles either way — because what gets
charged is whether the warp split at all, not how many lanes ended up on
each side. That table, and the `divergences`/`cycles` counters that produced
it, follow.

## How it works

A warp is not 32 independent threads; it is one instruction stream shared by
32 lanes, each with its own registers and an *active mask* saying whether it
participates in the current instruction. Ordinary code runs with every lane
active. A data-dependent `if` changes that: the hardware evaluates the
predicate per lane, then issues the `then` body with only the true lanes
active, then the `else` body with only the false lanes active — two passes
through the instruction stream instead of one, each masking off half the
warp. If every lane's predicate agrees, the second pass is skipped entirely
and the branch is free.

This is why divergence is a *pattern* problem, not a *proportion* problem.
Real hardware does not care whether 1 lane or 16 disagree with the rest — it
still has to issue both passes, because a pass exists to serve whichever
lanes need it, even one. That is the opposite of
[memory coalescing](memory-coalescing.md), where cost tracks how many
128-byte segments get touched, or [false sharing](false-sharing.md), where
cost tracks how many cache lines get invalidated — both scale with *how
much* memory moves. Divergence scales with *whether a split happened at
all*.

The simulator this page uses, `mlsys.sim.GPU`, does not watch program
counters directly — it flags a warp as divergent when its 32 lanes end up
issuing a **different number of global memory operations**, which is exactly
what a guarded write (`if (cond) out[i] = v;`, no `else`) produces: lanes
that took the branch have one more store than lanes that did not. That is a
real and common shape — masked scatter, a bounds check, a ReLU-style skip —
and [counting distinct predicate values per warp block](../tasks/sys-warp-divergence-branch-count/task.md)
is the pure-Python version of the same idea, while
[checking whether all lanes in a warp agree first](../tasks/gpu-warp-uniform-check-helper/task.md)
is the standard guard used to skip the expensive path when they do. A
same-cost `if`/`else` — both arms touching memory the same number of times —
does not move this particular counter, which is a real limitation of it, not
a subtlety of hardware; the instruction-issue view of the same branch is
counted directly, without that blind spot, by
[modelling the serialized issue count for a divergent branch](../tasks/gpu-model-serialized-issue-count-for-a-divergent-branch/task.md).

A related but distinct cost is a *data-dependent loop*, where lanes disagree
not on which branch but on how many iterations to run — the warp runs as
long as its slowest lane needs, wasting every already-finished lane's
remaining slots. That is measured separately in
[loop-count divergence penalty](../tasks/gpu-loop-count-divergence-penalty/task.md),
because it costs wasted iterations rather than a doubled instruction pass.

## Divergence and cycles measured against lane pattern

A one-warp kernel does a guarded write, `if (sel[t]) out[t] = v * 2.0;`,
where `sel` is a 0/1 array supplied per lane. Only the *pattern* of `sel`
changes across runs — the arithmetic each lane does is identical regardless
of pattern, so nothing but the branch outcome varies.

| lane pattern | lanes taking branch | divergences | cycles |
|---|---|---|---|
| none-take | 0/32 | 0 | 400 |
| all-take | 32/32 | **0** | 601 |
| half-take | 16/32 | 1 | 601 |
| alternating | 16/32 | 1 | 601 |
| one-lane | 1/32 | **1** | **601** |

Reproduce it — no NVIDIA hardware required, the `.cu` is executed by the
software GPU in [`src/mlsys/sim/gpu.py`](../src/mlsys/sim/gpu.py):

```bash
pip install mlsys-lab
python3 - <<'PY'
from mlsys.sim import CudaProgram, GPU
import numpy as np

SRC = """
__global__ void guarded_write(float* out, const float* in, const int* sel, int n) {
    int t = threadIdx.x;
    float v = in[t];
    if (sel[t]) {
        out[t] = v * 2.0;
    }
}
"""
N, BLOCK = 32, 32
patterns = {
    "none-take (0/32)":    [0]*32,
    "all-take (32/32)":    [1]*32,
    "half-take (16/32)":   [1]*16 + [0]*16,
    "alternating (16/32)": [1, 0]*16,
    "one-lane (1/32)":     [1] + [0]*31,
}
for name, sel in patterns.items():
    gpu = GPU(3 * N)
    gpu.gmem[0:N] = np.arange(N, dtype=np.float64)
    gpu.gmem[2*N:3*N] = np.array(sel, dtype=np.float64)
    m = CudaProgram(SRC).launch(gpu, "guarded_write", 1, BLOCK,
                                 {"out": N, "in": 0, "sel": 2*N, "n": N})
    print(f"{name:22s} divergences={m['divergences']} cycles={m['cycles']}")
PY
```

Read the middle two rows first: half-take and one-lane both report
**divergences=1 and cycles=601 — identical to each other**, even though one
has sixteen times as many lanes taking the branch as the other. The counter
does not know or care about the split ratio; it only knows the warp's lanes
disagreed. The more surprising row is all-take: every lane agrees, yet it
costs exactly what the divergent rows cost, because agreeing to *do* the
extra store still touches the write segment once. The only row that is
actually free is none-take, where no lane stores anything at all. The naive
expectation — "agreement is free, disagreement costs" — is half right: the
real fault line in this model runs between *nobody touches the segment* and
*somebody does*, not between uniform and divergent.

## Practise it

```bash
mlsys grade gpu-branchless-causal-mask-via-predicated-select
```

[That task](../tasks/gpu-branchless-causal-mask-via-predicated-select/task.md)
gates a real `.cu` on `max_abs_err <= 1e-06` and, separately,
`divergences == 0`. The shipped starter is a plain `if (j <= i) out[idx] =
score[idx]; else out[idx] = -1.0e30f;` causal mask — it gets every value
exactly right (`max_abs_err = 0.0`) and still fails, reporting
`divergences = 62`, because almost every one of the 64-wide row's warps
straddles the `j <= i` boundary and its lanes end up issuing a different
number of loads. **Correct values, divergent access pattern, failing
gate** — the fix folds the mask into arithmetic (`keep * v + (1-keep) *
neg_inf`) so every lane always loads and always stores.

In roughly increasing difficulty:
[count distinct branch ids per warp](../tasks/sys-warp-divergence-branch-count/task.md) (no CUDA, pure NumPy),
[check whether all lanes in a warp agree](../tasks/gpu-warp-uniform-check-helper/task.md) (the standard guard),
[predict per-warp divergence path count](../tasks/gpu-predict-per-warp-divergence-path-count/task.md) (write the predicate, the grader aggregates it),
[model the serialized issue count for a divergent branch](../tasks/gpu-model-serialized-issue-count-for-a-divergent-branch/task.md) (the instruction-count view, not the access-count view), and
[the loop-count divergence penalty](../tasks/gpu-loop-count-divergence-penalty/task.md) (a data-dependent trip count, not a branch).

## Common mistakes

- **Assuming cost scales with how many lanes disagree.** The table above
  shows one dissenting lane costing exactly what sixteen cost — 601 cycles
  either way. There is no partial credit for "mostly agreeing."
- **Gating the memory access instead of the value.** `if (cond) out[i] =
  v;` looks harmless but makes lanes issue different numbers of stores,
  which is precisely what this simulator's `divergences` counter — and real
  hardware's serialized second pass — charges for. Computing `v` for every
  lane and blending with a 0/1 predicate keeps the access pattern uniform.
- **Trusting `divergences == 0` to mean "no branches."** The counter only
  sees mismatched *global-memory-access counts*; a same-cost `if`/`else`
  where both arms touch memory identically often reports zero here even
  though it is a real branch, which is why the issue-count model in
  [gpu-model-serialized-issue-count-for-a-divergent-branch](../tasks/gpu-model-serialized-issue-count-for-a-divergent-branch/task.md)
  exists as a separate, complementary view.
- **Reading `cycles` as wall-clock time.** As with
  [memory coalescing](memory-coalescing.md), it is a deterministic count —
  transactions times a fixed latency — not a timer, which is why stride and
  lane pattern reproduce identically on every machine that runs the
  snippet.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[GPU-Puzzles](https://github.com/srush/GPU-Puzzles)** — the best-known
  resource in this space, 14 notebook puzzles building the same
  thread/warp intuition. It does not measure divergence, coalescing, or
  bank conflicts at all; it self-checks output only.
- **[LeetGPU](https://leetgpu.com/)** and **[Tensara](https://tensara.org/)**
  — browser judges on real GPUs, scored on correctness plus relative
  wall-clock speed. A divergent kernel shows up as "slower," never as a
  named, reproducible count, and the number depends on shared hardware.
- **[SW Online Judge](https://swforces.com/)** — real CUDA-C transpiled to
  OpenMP and run on CPU, the closest cousin to this page's no-GPU-needed
  approach. Its own documentation states it verifies correctness only, so a
  branchy-but-correct kernel passes there with no signal about the branch.
- **[GPU MODE lectures](https://github.com/gpu-mode/lectures)** — strong,
  actively maintained expository material covering warps and control flow
  in real depth. No submission or grading system; read it for the mental
  model, then come back here for the number.
- **[LeetCUDA](https://github.com/xlite-dev/LeetCUDA)** — 200+ reference
  kernels showing what divergence-free code looks like in practice
  (masking, predication) once puzzles stop being useful. Nothing to submit.

## References

1. NVIDIA, *CUDA C++ Programming Guide*, §7.1 "SIMT Architecture" — the
   active-mask and predicated-execution model this page's simulator
   approximates. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture
2. NVIDIA Developer Blog, *Using CUDA Warp-Level Primitives*. Covers the
   warp-synchronous model that makes divergence well-defined in the first
   place. https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
