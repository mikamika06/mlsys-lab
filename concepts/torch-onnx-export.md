---
title: "What is torch onnx export?"
description: "Torch onnx export explained, with a measured torch-vs-onnxruntime output difference and a silently-wrong branch you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is torch onnx export?

Torch ONNX export converts a trained PyTorch module into a static ONNX graph that
onnxruntime, TensorRT, or any other ONNX-compatible engine can run without PyTorch
installed. That conversion is not lossless: a two-layer network exported here comes back
from onnxruntime `1.192e-07` away from the torch output on the same input, and a model with
an `if` on a tensor's value can come back `1.7214` away — completely wrong — on an input
that never left PyTorch. Both are measured below, on models small enough to read in full.

## How it works

`torch.onnx.export` has two different implementations behind one function name, and which
one runs changes what "export" even means. The legacy path traces the model — runs it once
on example inputs and records every tensor operation that executed — then serializes that
fixed sequence of ops as the ONNX graph. The current default, since PyTorch 2.9, instead
calls `torch.export.export` to build an `ExportedProgram`: a graph captured through
`torch.compile`'s machinery, which understands shapes symbolically instead of just replaying
one execution.

The difference matters most exactly where PyTorch code is not a straight-line computation.
A trace has no notion of "this `if` could have gone the other way" — it only knows which
branch actually ran, and it bakes that branch in as if it were the only one that ever
existed. `torch.export`, by contrast, tries to reason about the condition symbolically and,
when it cannot prove which branch a future input will take, refuses to export at all rather
than guess. Neither behavior is a bug exactly — a traced graph is a valid description of the
one path it saw, and a refusal to export is safer than a wrong answer — but a reader who only
checks "did export succeed, and does the output match on my test input" will not notice the
trace silently commits to a branch until an input that needed the other one comes through in
production. This is the same category of error as [warp divergence](warp-divergence.md):
both are about what happens to hardware or software that expects one code path per thread
per instruction when the data disagrees at runtime, except here the failure is silent instead
of merely slow.

The rest of export is a translation problem: every PyTorch op the traced or captured graph
contains needs an ONNX operator, and not all of them have a one-to-one match — some
decompose into several ONNX primitives, the same "count what a name actually costs, not what
it promises" instinct as reading [GGUF vs safetensors](gguf-vs-safetensors.md) block metadata
instead of trusting the "4-bit" label. Numerically, ONNX graphs run in whatever dtype the
model was in when exported — a model exported in [bfloat16](bfloat16-vs-float16.md) keeps
bfloat16's coarser rounding all the way through onnxruntime, and floating-point operations
that reorder (a fused kernel on one side, several separate ops on the other, the same
non-associativity behind [Kahan summation](kahan-summation.md)) are exactly why "the same
model" produces a nonzero, if usually tiny, difference even when nothing is actually wrong.
Crossing the trace boundary at all is a serialization problem structurally similar to what
[Python's `multiprocessing`](python-multiprocessing.md) has to solve to send an object to
another process: both need every argument flattened into something the far side's fixed
interface can reconstruct.

## What torch vs onnxruntime disagree on

Two things were measured on a fixed random seed: how far a straight-line model's onnxruntime
output drifts from its torch output, and what a model with an `if` on `x.sum()` does once its
export is asked to run an input that should take the branch it never traced.

| model | export path | input relative to trace | max abs diff |
|---|---|---|---|
| 2-layer MLP, no branching | legacy trace (`dynamo=False`) | same input as export | 1.192e-07 |
| branch-on-sum model | legacy trace (`dynamo=False`) | same branch as export | 5.960e-08 |
| branch-on-sum model | legacy trace (`dynamo=False`) | **other branch** | **1.7214** |
| branch-on-sum model | `torch.export` (`dynamo=True`, current default) | either | export raises `TorchExportError` |

Reproduce it:

```bash
pip install mlsys-lab torch onnxruntime onnxscript
python3 - <<'PY'
import io
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

torch.manual_seed(0)

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SmallNet().eval()
x = torch.randn(4, 16)
buf = io.BytesIO()
torch.onnx.export(model, (x,), buf, input_names=["x"], output_names=["y"],
                   opset_version=17, dynamo=False)
buf.seek(0)
torch_out = model(x).detach().numpy()
sess = ort.InferenceSession(buf.read(), providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {"x": x.numpy()})[0]
plain_diff = float(np.abs(torch_out - onnx_out).max())
print(f"plain_max_abs_diff={plain_diff:.3e}")

class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x):
        if x.sum() > 0:
            return self.a(x)
        return self.b(x)

bmodel = BranchNet().eval()
x_pos, x_neg = torch.ones(1, 4), -torch.ones(1, 4)

buf2 = io.BytesIO()
torch.onnx.export(bmodel, (x_pos,), buf2, input_names=["x"], output_names=["y"],
                   opset_version=17, dynamo=False)
buf2.seek(0)
sess2 = ort.InferenceSession(buf2.read(), providers=["CPUExecutionProvider"])
onnx_pos = sess2.run(None, {"x": x_pos.numpy()})[0]
onnx_neg = sess2.run(None, {"x": x_neg.numpy()})[0]
traced_branch_diff = float(np.abs(bmodel(x_pos).detach().numpy() - onnx_pos).max())
wrong_branch_diff = float(np.abs(bmodel(x_neg).detach().numpy() - onnx_neg).max())
print(f"traced_branch_diff={traced_branch_diff:.3e}")
print(f"wrong_branch_diff={wrong_branch_diff:.4f}")

try:
    buf3 = io.BytesIO()
    torch.onnx.export(bmodel, (x_pos,), buf3, input_names=["x"], output_names=["y"],
                       opset_version=17, dynamo=True)
    print("dynamo_export=succeeded")
except Exception as e:
    print(f"dynamo_export_failed={type(e).__name__}")
PY
```

The first row is the number every tutorial shows and calls "correct": `1.192e-07`, float32
rounding noise from two independently-executed graphs computing the same arithmetic in a
different op order. The branch model traced and re-run on its own branch is even smaller,
`5.960e-08` — nothing about branching itself costs accuracy. The row that matters is the
third: the same exported file, asked for the branch it was never shown, returns an answer
`1.7214` away from the truth while reporting no error at all — `onnxruntime.InferenceSession`
has no way to know a branch was skipped. The current default export path catches exactly this
model and refuses to export it, trading a runtime silent-wrong-answer for a build-time
`TorchExportError` — which is safer, but only if nobody quietly passes `dynamo=False` to make
the error go away.

## Practise it

```bash
mlsys grade rwc-fix-a-wrong-transpose-perm-that-corrupts-the-export
```

[That task](../tasks/rwc-fix-a-wrong-transpose-perm-that-corrupts-the-export/task.md) gates
`fix_transpose_perm` on `perm_exact == 1.0` and `max_abs_err <= 1e-12`: given an input tensor,
a corrupted exported `Transpose`'s output, and the trusted torch reference, recover the
permutation the export should have used. The shipped starter returns the identity
permutation — the same silent assumption this page's branch model makes, that whatever the
export already computed must be the layout that was intended — so it fails both gates on
every case except a no-op transpose.

More of the same track, in increasing difficulty:
[map PyTorch ops to native ONNX ops vs. decompositions](../tasks/rwc-map-pytorch-ops-to-native-onnx-ops-vs-decompositions/task.md),
[flag export-incompatible ops (guard/break sources)](../tasks/rwc-flag-export-incompatible-ops-guard-break-sources/task.md),
[count Dynamo graph breaks in a traced function](../tasks/rwc-count-dynamo-graph-breaks-in-a-traced-function/task.md),
[run ONNX-style shape inference through dynamic axes](../tasks/rwc-run-onnx-shape-inference-through-dynamic-axes/task.md),
and [decompose SDPA into ONNX primitives and match fused attention](../tasks/rwc-decompose-sdpa-into-onnx-primitives-and-match-fused-attention/task.md).

## Common mistakes

- **Testing export only on the input you exported with.** The branch model's own-branch
  error is `5.960e-08` — indistinguishable from ordinary rounding noise — right up until an
  input that needed the untraced branch arrives, where the error jumps to `1.7214` with no
  warning from either PyTorch or onnxruntime.
- **Treating a successful export as proof of correctness.** The legacy `dynamo=False` path
  exported the branch model without complaint; only the newer `torch.export`-based path,
  `dynamo=True`, refuses, and only because it can prove the condition is data-dependent —
  it does not run the model on more inputs to check, it reasons about the trace it has.
- **Assuming `if`/`for` on a plain Python value is the same as on a tensor.** Branching on
  `len(x)` or a Python `int` is baked in safely because it cannot change between calls with
  the same shape; branching on `x.sum() > 0` depends on the data, which is exactly what a
  trace cannot see past.
- **Reading opset version as a formality.** `opset_version=17` above was silently upgraded to
  a newer version internally on this model; an operator missing from the requested opset is a
  translation failure, not a numeric one, and it surfaces as an export-time error rather than
  a wrong number.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which lists this
whole area as **115 tasks, adjacent only**:

- **[PyTorch official tutorials — torch.compile, torch.export, Dynamo](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)**
  is the canonical explanation of graph breaks, guards, and dynamic shapes, with runnable
  cells — but it is documentation to read and run, not a problem set with a checked answer.
- **[ONNX Tutorials](https://github.com/onnx/tutorials)** covers exporting from
  PyTorch/TensorFlow/scikit-learn and running the result on several runtimes — real
  run-and-read notebooks, no correctness checker on the numbers this page measures.
- **[ONNX Backend Test suite](https://onnx.ai/onnx/repo-docs/OnnxBackendTest.html)** is the
  closest thing to a graded operator-conformance resource, but it is built for people
  implementing an ONNX runtime backend, not for learners studying export from PyTorch.
- **[depyf](https://github.com/thuml/depyf)** decompiles the bytecode Dynamo generates back
  into readable Python, which is the tool to reach for while doing the graph-break tasks
  above, not a teaching resource by itself.
- **[MLC: Machine Learning Compilation](https://mlc.ai/courses.html)** teaches compiler
  internals through Apache TVM with graded notebooks — real grading, but a different stack
  from torch.compile/ONNX, useful only as background on why graph compilers behave this way.

## References

1. PyTorch, *torch.onnx* — exporter overview, the `dynamo` flag, and the
   TorchScript-to-`torch.export` migration. https://docs.pytorch.org/docs/stable/onnx.html
2. PyTorch tutorials, *Export a model with control flow to ONNX*.
   https://docs.pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
3. ONNX Runtime, *Python API documentation* — `InferenceSession` and provider selection.
   https://onnxruntime.ai/docs/api/python/api_summary.html
