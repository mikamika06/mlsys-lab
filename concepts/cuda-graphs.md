---
title: "What are CUDA graphs?"
description: "CUDA graphs explained, with a measured host-launch-count table (eager vs. one replay per iteration) and a re-capture count against how often the input shape changes, both reproducible with plain arithmetic, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What are CUDA graphs?

A CUDA graph is a capture-once, replay-many mechanism: a runtime records a fixed sequence of
GPU kernel launches into one graph object, then replays that sequence later with a single
host-side call instead of one call per kernel. Twenty small kernels run a thousand times cost
20,000 launches without a graph, 1,020 with one — measured below, no GPU involved.

## How it works

Every kernel launch, however tiny the kernel, costs the CPU a fixed slice of driver work:
validate arguments, resolve the grid/block configuration, push a command into the GPU's queue.
A step chaining dozens of small ops — a norm, a few matmuls, a softmax, some elementwise adds —
can finish each kernel faster than the CPU queues the next, so the step becomes launch-bound:
the CPU, not the GPU, sets the pace.

A CUDA graph removes that per-kernel cost from the steady state. `cudaStreamBeginCapture` records
every submitted operation instead of executing it; `cudaStreamEndCapture` freezes the recording
into a `cudaGraph_t`; `cudaGraphInstantiate` turns it into an executable `cudaGraphExec_t`; and
`cudaGraphLaunch` replays the whole sequence with one call. The N separate launches happen once,
during capture — every replay after is a single host call regardless of kernel count.

That speed comes with the constraint that decides whether graphs apply at all: **a captured
graph's kernel arguments — shapes, strides, and device pointers — are frozen at capture time.**
Replay does not re-resolve indices or re-check sizes; it reissues the same kernels against the
same memory addresses recorded during capture. Feed it a differently-sized batch, or a tensor
the allocator placed elsewhere, and the graph either reads garbage or must be re-captured from
scratch. This is why graphs pair naturally with a decode loop over a [KV cache](kv-cache.md)
padded to a fixed bucket, and badly with one whose sequence length genuinely grows every step,
or with a [continuous-batching](continuous-batching.md) scheduler whose active batch size
changes on every admission and eviction.

### PyTorch's `torch.cuda.graph` API

PyTorch wraps the raw calls in `torch.cuda.graph`, a context manager over the lower-level
`torch.cuda.CUDAGraph` object. The pattern: allocate a *static* input tensor once, warm up a few
iterations on a side stream (capture is picky about unsettled allocator memory), then capture
the real call inside `with torch.cuda.graph(g):`. Every step after that, copy new data into the
static input in place and call `g.replay()`. The output tensor is just as static, and
[replaying a captured static-buffer graph correctly](../tasks/rwc-replay-a-cuda-graph-static-buffer-capture-correctly/task.md)
is built around the consequence: copy out `g.replay()`'s result before the next call overwrites
the one output buffer every replay shares.

### `torch.compile` and dynamic shapes

`mode="reduce-overhead"` in [`torch.compile`](torch-compile.md) applies CUDA-graph capture
automatically to whatever region Dynamo compiles, instead of a hand-written capture/replay
dance. The frozen-shape constraint follows it under a different name: Dynamo recompiles a
region when a new input's shape, dtype, or rank fails a guard — the same event
[modeled directly as a guard check](../tasks/sys-generate-recompile-guards/task.md) — and under
`reduce-overhead` that failure forces a fresh graph capture on top of the retrace, paying both
costs together. `dynamic=True` widens Dynamo's guards to tolerate a *range* of shapes without
retracing the FX graph, but cannot widen a captured graph's frozen buffers the same way.

## Launches measured against iteration count

Fixing twenty kernels per iteration — a plausible op count for one small transformer block —
and varying only the iteration count, the script counts host-side launch calls two ways: `N`
per iteration without a graph, one capture (`N` launches) plus one replay per iteration with a
graph.

| iterations | eager launches | graph launches | reduction |
|---|---|---|---|
| 1 | 20 | 21 | **0.95x** |
| 2 | 40 | 22 | 1.82x |
| 5 | 100 | 25 | 4.00x |
| 20 | 400 | 40 | 10.00x |
| 100 | 2,000 | 120 | 16.67x |
| 1,000 | 20,000 | 1,020 | 19.61x |
| 10,000 | 200,000 | 10,020 | **19.96x** |

Reproduce it — pure arithmetic, no simulator or GPU required:

```bash
pip install mlsys-lab
python3 - <<'PY'
N = 20  # kernels per iteration, e.g. one small transformer decode step
for iters in (1, 2, 5, 20, 100, 1000, 10000):
    eager = N * iters                 # one host launch per kernel, every iteration
    graph = N + iters                 # N launches to capture once, then 1 replay/iteration
    ratio = eager / graph
    print(f"iters={iters} eager={eager} graph={graph} reduction={ratio:.2f}x")
PY
```

Read the last column as the argument for graphs, and the first row as its warning label. At a
single iteration the graph is **worse than doing nothing** — 21 launches against 20 — because
capture pays the full N launches up front with no replay left to amortize it. The reduction
crosses 1.00x only once a second iteration exists to spend that cost on, and it climbs toward,
but never reaches, N=20 — the ceiling set by capture happening exactly once regardless of how
many cheap replays follow.

## Practise it

```bash
mlsys grade rwb-launch-overhead-step-time-model
```

[That task](../tasks/rwb-launch-overhead-step-time-model/task.md) gates
`graph_launch_step_time(L, N, C)` on relative L2 error `< 1e-9` against
`[N*L + C, L + C, (N-1)*L / (N*L + C)]`, the same split above generalized to a per-launch cost
`L` and compute term `C` this page's model leaves out. The failure it catches: computing the
fraction removed as `N*L / eager_time`, assuming the graph's own launch is free — fine for
large `N`, measurably wrong once `N` is small enough for that one launch to matter, exactly row
one's gap above.

Two more angles, real `.cu` kernels, no graph API involved:
[find the batch size where a launch's fixed cost stops dominating a kernel's own compute](../tasks/gpu-model-launch-overhead-amortization/task.md),
and
[compare relaunching a kernel K times against one persistent kernel looping K times internally](../tasks/gpu-persistent-kernel-vs-relaunch-tradeoff/task.md).

## Recaptures measured against how often the shape changes

The frozen-shape constraint above has a direct cost: a workload whose shape changes every `k`
steps forces a fresh N-launch capture at the start of each segment, and only the remaining
`k-1` steps get the cheap one-launch replay. Fixing 1,000 total steps and 20 kernels per step,
the only thing varied is `k`.

| recapture every k steps | recaptures | launches (graphs) | reduction |
|---|---|---|---|
| 1 | 1,000 | 20,000 | **1.00x** |
| 2 | 500 | 10,500 | 1.90x |
| 4 | 250 | 5,750 | 3.48x |
| 8 | 125 | 3,375 | 5.93x |
| 16 | 63 | 2,197 | 9.10x |
| 32 | 32 | 1,608 | 12.44x |
| 64 | 16 | 1,304 | 15.34x |
| 200 | 5 | 1,095 | 18.26x |
| 1,000 | 1 | 1,019 | **19.63x** |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import math

N, T = 20, 1000             # kernels per step, total steps
eager_total = N * T          # unaffected by k -- eager never captures anything
for k in (1, 2, 4, 8, 16, 32, 64, 200, 1000):
    recaptures = math.ceil(T / k)                          # one capture per shape segment
    launches = recaptures * N + (T - recaptures) * 1        # N per capture step, 1 per replay
    ratio = eager_total / launches
    print(f"k={k} recaptures={recaptures} launches={launches} reduction={ratio:.2f}x")
PY
```

At `k=1` — a new shape every step, what an unbucketed batch size or growing sequence length
produces — every step is a capture step, `launches` equals `eager_total`, and the graph buys
**nothing**: 1.00x is no optimization at all. Only once shapes hold for more than a couple of
steps does the curve climb, needing `k=1,000` to approach the ~19.6x ceiling the previous table
found. That is the case for
[rounding a batch up to the nearest pre-captured bucket](../tasks/rwb-round-a-batch-up-to-the-nearest-captured-bucket/task.md)
rather than capturing the exact size on demand, and for
[treating any batch past the largest bucket as an eager fallback](../tasks/rwb-classify-captured-vs-eager-fallback-batches/task.md)
instead of a fresh capture.

## Common mistakes

- **Assuming one capture covers every future call.** A batch sized differently from capture
  makes replay wrong or undefined — no shape check exists to fall back on, since removing
  runtime checks was the whole point.
- **Reading a replay's output after the next replay runs.** The output tensor is the same
  memory every call; keeping a reference instead of copying it out gets silent corruption,
  exactly what
  [the static-buffer replay task](../tasks/rwc-replay-a-cuda-graph-static-buffer-capture-correctly/task.md)
  is built to catch.
- **Treating `dynamic=True` as a fix for capture cost.** It stops Dynamo retracing the FX graph
  across a shape range; it does nothing for a `reduce-overhead` graph's frozen buffers, which
  still need one capture per distinct shape seen.
- **Judging the win at one fixed iteration count.** The first table's row one is 5% *slower*
  than eager at a single iteration — the payoff depends on replay count, not a constant you can
  quote alone.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which files CUDA
graphs under "Compilation and export" — 115 tasks in this bank, and every other resource for
that whole area is documentation, a debugging tool, or reference code, adjacent rather than
overlapping:

- **NVIDIA's own CUDA Graphs documentation and PyTorch's integration guide**, references 1-3
  below, remain the best explanation of capture semantics — prose and snippets, nothing that
  counts a launch or a recapture.
- **[GPU MODE / KernelBot](https://www.gpumode.com/)** and **[LeetGPU](https://leetgpu.com/)** —
  real or CPU-emulated GPU judges scored on correctness and speed. Either can run a graph-backed
  kernel, but neither reports a launch or capture count as a first-class number.
- **[depyf](https://github.com/thuml/depyf)** decompiles what Dynamo traced — reaches for *why*
  `mode="reduce-overhead"` re-captured, where this page only counts *how many times*.
- No resource found turns "how many host launches does this issue" or "how many recaptures a
  shape-changing workload forces" into a number you check yourself — the gap the tables and
  tasks above fill.

## References

1. NVIDIA, *CUDA Graphs documentation & PyTorch integration guide*.
   https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/introduction.html
2. NVIDIA, *CUDA C++ Programming Guide*, §"CUDA Graphs".
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
3. PyTorch documentation, *CUDA semantics*.
   https://docs.pytorch.org/docs/stable/notes/cuda.html
4. Gray, A., NVIDIA Developer Blog, *Getting Started with CUDA Graphs*, 2019.
   https://developer.nvidia.com/blog/cuda-graphs/
