<p align="center">
  <img src=".github/assets/logo.png" alt="mlsys-lab" width="120">
</p>

<h1 align="center">mlsys-lab</h1>

<p align="center">
  <em>Auto-graded exercises in low-level ML systems: quantization, KV cache and attention,<br>
  batching, numerics, CPU performance, CPython internals, GPU memory behaviour.<br>
  Mostly Python, with real C++ and real CUDA-C. A measured number is the gate, never a clock.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/mlsys-lab/"><img src="https://img.shields.io/pypi/v/mlsys-lab?logo=pypi&logoColor=white&label=pypi" alt="PyPI"></a>
  <a href="https://open-vsx.org/extension/mikamika06/mlsys-lab"><img src="https://img.shields.io/open-vsx/v/mikamika06/mlsys-lab?logo=visualstudiocode&logoColor=white&label=extension" alt="Extension"></a>
  <a href="https://github.com/mikamika06/mlsys-lab/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mikamika06/mlsys-lab/ci.yml?branch=main&label=ci" alt="CI"></a>
  <a href="https://github.com/mikamika06/mlsys-lab/actions/workflows/bank.yml"><img src="https://img.shields.io/github/actions/workflow/status/mikamika06/mlsys-lab/bank.yml?branch=main&label=task%20bank" alt="Task bank"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>
</p>

<!-- ─────────────────────────────────────────────────────────────────────────────
     DEMO GOES HERE.

     Record `mlsys start` then `mlsys grade` on the bank-conflict task — the
     16.5 dropping to 1.0 is the whole pitch in five seconds. Drop the file in
     .github/assets/ and swap the <img> in below.

     GIF, not mp4: of twenty popular dev-tool READMEs measured, six embed a GIF
     and NOT ONE embeds a video, because GitHub will not play one inline. Record
     with asciinema + agg, or vhs (github.com/charmbracelet/vhs).

<p align="center">
  <img src=".github/assets/demo.gif" alt="Grading a CUDA task: max_abs_err passes, smem_wave_ratio fails at 16.5, then passes at 1.0" width="760">
</p>
     ───────────────────────────────────────────────────────────────────────── -->

```
$ mlsys grade gpu-1-padding-to-remove-transpose-bank-conflicts

  Kill transpose bank conflicts with +1 padding

  ✓ max_abs_err                 0   gate <= 1.000e-09
  ✗ smem_wave_ratio          16.5   gate <= 1.05

  ● FAIL  1/2 gates
```

Your transpose is numerically perfect and 16.5× slower than it needs to be, because
the tile is 32 floats wide and every column read lands in one shared-memory bank.
Nothing was timed to find that out.

```bash
pip install mlsys-lab
mlsys start gpu-1-padding-to-remove-transpose-bank-conflicts   # writes the starter here
mlsys grade gpu-1-padding-to-remove-transpose-bank-conflicts   # measure it
```

No cloud, no account, no GPU. Every metric is deterministic, so the same submission
scores identically on any machine.

Fourteen areas, from the CPython data model to paged attention. A survey of 135 other
resources found that **six of those fourteen have no automatically graded material
anywhere** — the survey is in [`RESOURCES.md`](RESOURCES.md), and it says where other
people teach something better than this does.

## Projects

A task is one function. A project is a repo: a ticket that names a symptom rather
than the defect, several files to edit, and milestones graded one at a time so
progress is visible before the whole thing is finished.

| | task | project |
|---|---|---|
| you edit | one file | 4–8 files in a real layout |
| the statement | names the defect | names the symptom |
| the gate | a measured number | an invariant, a ratio against your own baseline, or a recorded run |
| finishing | pass or fail | 7 milestones, each with its own gates |

The last milestone of every project is the same shape: you write a regression test,
we inject a fault into your own code, and your test has to catch it.

`mlsys.sim.server` makes this work without hardware — a deterministic model of
continuous batching over a paged KV cache (reference-counted blocks, chunked
prefill, prefix caching, preemption by recompute or swap) on an integer clock, so
a scheduler you write scores the same on any machine.

## Why the numbers are trustworthy

Wall-clock timing is never a gate. Instead the engine models the hardware and
counts what actually happened:

| you write | it runs on | it measures |
|---|---|---|
| `solve.cu` | a software GPU: 32-lane warps, 128-byte transactions, 32 shared-memory banks, barriers, warp shuffles | coalescing transactions, bank-conflict waves, warp divergence, cycles |
| `solve.cpp` | the local `clang++ -O2 -std=c++20`, plus a modelled set-associative cache | cache misses, lines touched, and the program's own output |
| `solve.py` | the interpreter, with an instrumented probe | allocations, copies, numerical error against a computed oracle |

The reference answer is always **computed by the grader**, never hard-coded — so
a task cannot be passed by guessing the expected output.

## Install

[PyPI](https://pypi.org/project/mlsys-lab/) ·
[VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=mikamika06.mlsys-lab) ·
[Open VSX](https://open-vsx.org/extension/mikamika06/mlsys-lab)

```bash
pip install mlsys-lab                             # the engine and the whole task bank

mlsys list                                        # browse the bank
mlsys start gpu-ex-cuda-coalesced-scale           # writes ./<task-id>/solve.cu to edit
mlsys run   gpu-ex-cuda-coalesced-scale           # execute it, see what it does
mlsys grade gpu-ex-cuda-coalesced-scale           # measure it

mlsys project list                                # multi-file projects
mlsys project start p-continuous-batching-scheduler
mlsys project grade p-continuous-batching-scheduler --milestone 1
```

```
  ✗ max_abs_err          9.632   gate <= 1.000e-09
  ✓ transactions             0   gate <= 20

  ● FAIL  1/2 gates
```

The bank ships inside the package, so there is nothing to clone and nothing to
download on first run. `mlsys start` copies the task's starter into a directory of
yours; `mlsys run` and `mlsys grade` then find it. All three work the same for
Python, C++ and CUDA tasks — the file extension follows the task. The bank itself
is never written to, so your work and the tasks never mix.

`run` and `grade` answer different questions. Grading applies the gates and is the
only thing that marks a task solved; running just executes the file — the script,
or `clang++` against the task's own driver, or the kernel on the software GPU —
and shows the print, the traceback and the compiler diagnostic that a verdict
deliberately hides.

C++ tasks need a C++20 compiler (`clang++` or `g++`); everything else is Python
plus NumPy. CUDA tasks need **no** NVIDIA hardware — the `.cu` is executed by the
simulator in `src/mlsys/sim/`.

Sixteen tasks have an oracle that needs more than NumPy — `scipy`, `ml-dtypes`,
`mpmath`, or `torch`. Rather than make every learner install torch, each declares what
it needs and `mlsys grade` tells you:

```bash
pip install "mlsys-lab[extras]"     # scipy, ml-dtypes, mpmath — 15 of the 16
pip install "mlsys-lab[torch]"      # the remaining one
```

Five tasks use ARM NEON intrinsics and one loads Apple's Accelerate framework, so they
only build on the platform they target. The verifier skips those by name rather than
pretending they passed.

### VS Code

Install **mlsys-lab** from the Marketplace, then run **mlsys-lab: Open Workspace**.
If the package is not installed yet, the extension offers to do it. Solutions go to
`~/mlsys-lab/<task-id>/` (configurable via `mlsys.workDir`).

### Contributing to the bank

```bash
git clone https://github.com/mikamika06/mlsys-lab
cd mlsys-lab && pip install -e .
```

A checkout takes precedence over the installed copy, so the tasks you are editing
are the ones that get graded. `$MLSYS_TASKS` overrides both.

## What is in the bank

Three native tracks plus the systems material:

- **Deep Python** — the data model, descriptors, the GIL, memory layout
- **Deep C++** — ownership, moves, ABI, undefined behaviour you can observe
- **CPU performance** — cache blocking, layout (AoS/SoA), branches, SIMD
- **GPU / CUDA** — coalescing, bank conflicts, divergence, barriers, shuffles
- **Numerics & tensors** — stability, shapes and strides, low-precision formats
- **LLM internals** — attention, RoPE, GQA, the KV cache, sampling
- **LLM systems** — paged KV, continuous batching, parallelism, speculative decoding
- **Applied** — quantization, attention/KV, export & compilation, serving,
  memory & offload, sparsity & distillation

## Concepts, explained with a number

[`concepts/`](concepts/) — 37 pages, one per concept, each with a measurement produced by
this repo and the command that regenerates it, then the graded exercise. Two checks run in
CI: one for structure and links, one that **executes each page's own snippet** and fails if
it prints anything the page does not say.

Start with [memory coalescing](concepts/memory-coalescing.md) (128-byte transactions against
read stride, 2 → 33), [false sharing](concepts/false-sharing.md) (coherence invalidations
against padding, 7,999 → 0), or [gguf vs safetensors](concepts/gguf-vs-safetensors.md)
(bytes per weight *actual* rather than the nominal 4 bits).

## Where else to practise this

[`RESOURCES.md`](RESOURCES.md) — 135 other places to practise the same material, in one list,
grouped by whether they check your work. Of the 135, **19 give you an automatic verdict** and
the rest do not; each entry says what it is, what it costs, and whether it is still alive.

[`LANDSCAPE.md`](LANDSCAPE.md) is the same 135 organised by this bank's own areas, with a
verdict on each: it surveys the other resources aimed at each of
these areas — 141 of them, every link checked — and marks whether each one actually
grades your work or only shows you code. It says where this bank is *not* your best
option, which is most of the point of having the page.

The short version: GPU/CUDA, algorithms-from-scratch and LLM internals are well served
already (GPU-Puzzles, LeetGPU, Tensara, deep-ml, CS336, LeetCUDA). Six of the fourteen
areas have **no** auto-graded resource anywhere, and four — applied quantization,
compilation & export, memory & offload, and sparsity/pruning/distillation — have nothing
to practise against at all: papers, production libraries and documentation, but no
exercise that tells you that you got it wrong.

## Writing a task

A task is a directory under `tasks/`. See [`TASK_FORMAT.md`](TASK_FORMAT.md).

```
tasks/<id>/
  meta.json     id, title, difficulty, gates       ("native": "cpp" | "cuda")
  task.md       ## Context / ## Task / ## Example / ## What the gate checks
  check.py      grade(sol, fx) -> {metric: value}  — computes its own reference
  starter.py    the empty contract the learner starts from
  solution_ref.py
```

```bash
tools/verify_task.sh   <id>    # python task:  reference passes, starter fails
tools/verify_native.sh <id>    # C++ / CUDA task
```

Both print `TASK_OK` only if the reference passes the gates **and** the shipped
starter fails them — a task that anything can pass is not a task.

## Layout

```
src/mlsys/        the engine
  sim/            software GPU, CUDA-C front end, cache model, ABI model
  runners/        cpp.py (clang++), cuda.py (software GPU)
  scorers.py      the metrics tasks gate on
  task_list2.json the curriculum: 14 areas, 291 sub-tracks
tasks/            the bank
concepts/         one page per concept, each with a measured number
editor/           VS Code extension
tools/            verify, queue helpers, browser preview
```

## License

MIT
