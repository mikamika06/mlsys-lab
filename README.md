# mlsys-lab

Auto-graded exercises in low-level ML systems. You write **real C++**, **real
CUDA-C** and Python; a local grader runs it and checks a **measured number**
against declarative gates.

No cloud, no account, no GPU required. Every metric is deterministic, so the same
submission scores identically on any machine.

```
$ mlsys grade gpu-1-padding-to-remove-transpose-bank-conflicts

  max_abs_err        0.0        <= 1e-12   PASS
  smem_wave_ratio    16.5       <= 1.05    FAIL
                     ^ your tile is 32 wide, so every column read hits one bank
```

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
git clone https://github.com/mikamika06/mlsys-lab
cd mlsys-lab
pip install -e .

mlsys list                       # browse the bank
mlsys grade <task-id>            # grade your attempt
```

C++ tasks need a C++20 compiler (`clang++` or `g++`); everything else is Python
plus NumPy. CUDA tasks need **no** NVIDIA hardware — the `.cu` is executed by the
simulator in `src/mlsys/sim/`.

### VS Code

```bash
cd editor && npx @vscode/vsce package && code --install-extension mlsys-lab-0.1.0.vsix
```

Open the repository folder and run **mlsys-lab: Open Workspace**.

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

## Writing a task

A task is a directory under `tasks/`. See [`docs/TASK_FORMAT.md`](docs/TASK_FORMAT.md).

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
tasks/            the bank
editor/           VS Code extension
tools/            verify, queue helpers, browser preview
docs/             curriculum and research
```

## License

MIT
