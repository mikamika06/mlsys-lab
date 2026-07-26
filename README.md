# mlsys-lab

Auto-graded exercises in low-level ML systems. You write **real C++**, **real
CUDA-C** and Python; a local grader runs it and checks a **measured number**
against declarative gates.

No cloud, no account, no GPU required. Every metric is deterministic, so the same
submission scores identically on any machine.

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

```bash
pip install mlsys-lab                             # the engine and all 2052 tasks

mlsys list                                        # browse the bank
mlsys start gpu-ex-cuda-coalesced-scale           # writes ./<task-id>/solve.cu to edit
mlsys grade gpu-ex-cuda-coalesced-scale           # measure it
```

```
  ✗ max_abs_err          9.632   gate <= 1.000e-09
  ✓ transactions             0   gate <= 20

  ● FAIL  1/2 gates
```

The bank ships inside the package, so there is nothing to clone and nothing to
download on first run. `mlsys start` copies the task's starter into a directory of
yours; `mlsys grade` then finds it. Both work the same for Python, C++ and CUDA
tasks — the file extension follows the task. The bank itself is never written to,
so your work and the tasks never mix.

C++ tasks need a C++20 compiler (`clang++` or `g++`); everything else is Python
plus NumPy. CUDA tasks need **no** NVIDIA hardware — the `.cu` is executed by the
simulator in `src/mlsys/sim/`.

### VS Code

Install **mlsys-lab** from the Marketplace, then run **mlsys-lab: Open Workspace**.
If the package is not installed yet, the extension offers to do it. Solutions go to
`~/mlsys-lab/<task-id>/` (configurable via `mlsys.workDir`).

PyPI: <https://pypi.org/project/mlsys-lab/> · Marketplace:
<https://marketplace.visualstudio.com/items?itemName=mikamika06.mlsys-lab> · Open VSX:
<https://open-vsx.org/extension/mikamika06/mlsys-lab>

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

[`concepts/`](concepts/) — one page per concept, each with a measurement produced by
the simulator in this repo and the command that regenerates it, then the graded exercise.
Start with [memory coalescing](concepts/memory-coalescing.md) (128-byte transactions
against read stride, 2 → 33) or [false sharing](concepts/false-sharing.md) (coherence
invalidations against padding, 7,999 → 0).

## What else exists

[`LANDSCAPE.md`](LANDSCAPE.md) surveys the other resources aimed at each of
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
