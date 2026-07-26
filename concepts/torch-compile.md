---
title: "What is torch compile?"
description: "Torch compile explained, with a measured graph-break count and reason string per function (clean, printing, tensor-branching, numpy) you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is torch compile?

Torch compile is PyTorch's just-in-time graph compiler: wrap a function in `torch.compile`
and Dynamo traces its bytecode into an FX graph that a backend turns into fused kernels
instead of running eager op by op. One `print()` call inside that function is enough to
split a single graph into 2, because Dynamo cannot represent it and falls back to plain
Python for that statement. Below is the exact graph count `torch._dynamo.explain()` reports
for four small functions, including one break that common write-ups get wrong.

## How it works

`torch.compile` is the user-facing entry point over two separable subsystems: Dynamo, which
intercepts Python bytecode and traces the tensor operations it sees into an FX graph, and a
backend — Inductor by default — that lowers that graph into Triton or C++ kernels. Dynamo's
guarantee is narrower than it sounds: it does not compile your function, it compiles *the
parts of your function it can represent as a static dataflow graph*, and falls back to plain
CPython for everything else. When it hits something it cannot represent — a `print`, a call
into a C extension it does not model, a branch whose condition depends on a tensor's runtime
value — it ends the current graph, executes that one operation eagerly, and starts a new
graph on the far side. Each such event is a *graph break*, and `torch._dynamo.explain(fn)(x)`
returns the resulting graph count, break count, and a human-readable reason for every break
found.

This matters because graph breaks are not cosmetic. Every one loses cross-op fusion at that
boundary: the fusion Inductor would apply across a [softmax](softmax-vs-sigmoid.md)-shaped
reduction, or across the mean/variance pass behind
[RMSNorm vs LayerNorm](rmsnorm-vs-layernorm.md), stops at the break and resumes as two
independently-compiled or eager pieces. It also multiplies recompilation risk: each subgraph
carries its own *guards* — runtime checks on shape, dtype, and any Python value the trace
depended on — and a guard failure on one subgraph retraces only that subgraph, not the whole
function. Fusing correctly is a code-generation problem too: the Triton kernels Inductor
emits still have to issue [coalesced](memory-coalescing.md) global loads and avoid
[warp divergence](warp-divergence.md) and
[bank conflicts](cuda-shared-memory-bank-conflicts.md) inside a block, exactly like a
hand-written kernel — Inductor's autotuner just searches those choices instead of a human
doing it. And because Inductor is free to reorder a fused reduction to keep it on one kernel,
a compiled sum does not always accumulate in the same order eager does, which is the same
ordering sensitivity behind [Kahan summation](kahan-summation.md).

None of this raises an error. A function with graph breaks still runs and returns the correct
tensor — it is only slower and more recompile-prone than the fully-fused version, silently,
which is exactly why counting breaks beats guessing at them.

## Graph breaks measured, not guessed

Four small functions, one call to `torch._dynamo.explain()` each: what varied is only the
body (a clean elementwise chain, one with a `print`, one branching on a tensor's value, one
calling `numpy` on a `.numpy()`-view of the tensor), and what was counted is the graph count,
the break count, and the first reported break reason.

| function | graph_count | graph_break_count | reported reason |
|---|---|---|---|
| clean | 1 | 0 | none |
| printing | 2 | 1 | Failed to trace builtin operator (`print`) |
| data_dependent | 2 | 1 | generic_jump TensorVariable() |
| numpy_call | **1** | **0** | none |

Reproduce it:

```bash
pip install mlsys-lab torch
python3 - <<'PY'
import torch
import torch._dynamo as dynamo

def clean(x):
    return (x * 2 + 1).sin()

def printing(x):
    y = x * 2
    print("y computed")
    return y + 1

def data_dependent(x):
    if x.sum() > 0:
        return x * 2
    return x - 2

def numpy_call(x):
    y = x * 2
    import numpy as np
    z = y.numpy()
    w = np.sum(z)
    return y + w

x = torch.randn(8)
for name, fn in [("clean", clean), ("printing", printing),
                  ("data_dependent", data_dependent), ("numpy_call", numpy_call)]:
    dynamo.reset()
    eo = dynamo.explain(fn)(x)
    reason = eo.break_reasons[0].reason.strip().splitlines()[0] if eo.break_reasons else "none"
    print(f"{name}: graph_count={eo.graph_count} graph_break_count={eo.graph_break_count} reason={reason}")
PY
```

Two things the table says that folklore does not. First, `print` and a data-dependent branch
on a *tensor* value both cost exactly 1 break here, but for different reasons Dynamo reports
verbatim — "Failed to trace builtin operator" for `print`, "generic_jump TensorVariable()" for
the branch — and that string, not a guess, is what you search for when a real model breaks.
Second, `numpy_call` costs 0 breaks on this torch build: Dynamo has native support for tracing
`.numpy()` and common `numpy` functions on a tensor-backed array, so `np.sum` is captured
inside the same graph as `<Wrapped function <original sum>>` rather than falling back to real
CPU numpy. "Calling into numpy always breaks the graph" is the kind of claim this page's rule
exists to catch — it was true in earlier PyTorch versions and is not true here, which is
exactly why the reproduction command matters more than the sentence.

## Practise it

```bash
mlsys grade rwc-count-dynamo-graph-breaks-in-a-traced-function
```

[That task](../tasks/rwc-count-dynamo-graph-breaks-in-a-traced-function/task.md) gates
`count_breaks_and_subgraphs` on `exact_match == 1.0` against seven fixed event traces, using
`events: list[str]` as a stand-in for Dynamo's own break stream — a `break_`-prefixed entry is
a graph break, everything else is a traced op. The shipped starter raises `NotImplementedError`
and fails the first case immediately. The harder failure mode is counting subgraphs as the
number of non-break *events* instead of the number of contiguous non-break *blocks*:
`["op", "op2"]` is one subgraph of two ops, `(0, 1)`, and a per-event counter reports `(0, 2)`
instead, inventing a break-free graph Dynamo never actually produced.

More of the same, in increasing difficulty:
[classify code snippets as traceable or a graph break by syntax alone](../tasks/sys-classify-graph-breaks/task.md),
[flag the specific ops `torch.export` treats as guard-break sources](../tasks/rwc-flag-export-incompatible-ops-guard-break-sources/task.md),
[decide whether a new input's shape, dtype, or rank forces a recompile](../tasks/sys-generate-recompile-guards/task.md),
[constant-fold a traced graph the way Inductor does before codegen](../tasks/sys-constant-fold-a-traced-graph/task.md),
and [dead-code-eliminate an FX-like graph down to what its outputs actually need](../tasks/sys-dead-code-elimination-on-fx-like-graph/task.md).

## Common mistakes

- **Trusting "no numpy in compiled code" as a rule.** The table's own `numpy_call` row scores
  0 graph breaks on this torch build; the correct question is always "what does
  `explain()` say on my version", not a folklore list.
- **Confusing `graph_count` with `graph_break_count`.** A function with 1 break produces 2
  graphs, not 1 — the break count is one less than the graph count, and reading them as the
  same number misreports how fused the result actually is.
- **Treating a graph break as an error.** All four functions above return a correct tensor;
  `printing` and `data_dependent` are simply split into 2 graphs instead of 1 and lose fusion
  at the seam, with no exception raised anywhere.
- **Debugging by staring at the traceback instead of the reason string.** `explain()`'s
  `break_reasons` names the exact cause — "Failed to trace builtin operator" versus
  "generic_jump TensorVariable()" point at completely different fixes, and neither shows up in
  a normal Python stack trace.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which lists this
whole area as **115 tasks in this bank, adjacent only** — meaning everything found is
documentation, a debugging tool, or reference code, not something with a starter to fail
against:

- **[PyTorch official tutorials: torch.compile, torch.export, troubleshooting & Dynamo deep-dive](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)**
  is the canonical explanation, and calls the graph break "one of the most fundamental
  concepts within `torch.compile`" — runnable cells, no grading, and it stops well short of
  guard/recompilation mechanics.
- **[depyf](https://github.com/thuml/depyf)** decompiles the bytecode Dynamo generates back
  into readable Python, so a break you only see as a count here can be located in the actual
  generated source — the tool you reach for after this page's exercise, not a substitute for
  it.
- **[MLC: Machine Learning Compilation](https://mlc.ai/courses.html)** is adjacent, not
  overlapping: a real course with graded notebooks, but built on Apache TVM/TensorIR rather
  than Dynamo/Inductor, and its content has not been revised since 2022.
- No resource found — including the two above — turns "does your fix reduce graph breaks or
  stop a recompile" into a deterministic pass/fail number the way the tasks linked above do.

## References

1. PyTorch, *Introduction to `torch.compile`*, intermediate tutorial.
   https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
2. PyTorch, `torch/_dynamo/eval_frame.py` — source of `explain()` and the
   `graph_count`/`graph_break_count`/`break_reasons` fields this page reads, tag v2.13.0.
   https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_dynamo/eval_frame.py
3. Yao, K. et al., *depyf: Open the Opaque Box of TorchDynamo*, JMLR (MLOSS), source at
   https://github.com/thuml/depyf
