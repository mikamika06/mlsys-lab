# Task format

A task is one directory under `tasks/`, named exactly as its `meta.json` `id`.
Three flavours share the same shape; only the language and the runner differ.

## Common

| file | role |
|---|---|
| `meta.json` | identity, difficulty, and the **gates** |
| `task.md` | the statement: `## Context` / `## Task` / `## Example` / `## What the gate checks` |

`meta.json`:

```json
{
  "id": "gpu-1-padding-to-remove-transpose-bank-conflicts",
  "title": "Kill transpose bank conflicts with +1 padding",
  "difficulty": 4,
  "genre": "optimize",
  "native": "cuda",
  "gates": [
    { "metric": "max_abs_err",     "op": "<=", "threshold": 1e-12 },
    { "metric": "smem_wave_ratio", "op": "<=", "threshold": 1.05 }
  ]
}
```

Every gate `metric` **must** be a key the grader returns. `native` is absent for
Python tasks, `"cpp"` for real C++, `"cuda"` for real CUDA-C.

## Python task

```
check.py          def grade(sol, fx) -> dict
starter.py        imports + signature + `raise NotImplementedError('your code here')`
solution_ref.py   passes every gate
gen_fixtures.py   optional, deterministic fixtures
```

`check.py` **computes its own reference** — with NumPy, or by running a model in
`mlsys.sim`. It never hard-codes an expected value, seeds every RNG, and wraps
each call into the learner's code in `try/except` so a crash becomes a failing
metric rather than a traceback.

```python
from mlsys import scorers
from mlsys.sim import cache

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    x = rng.random(4096)
    ref = np.cumsum(x)                 # the oracle, computed here
    try:
        got = sol.prefix_sum(x)
    except Exception:
        return {"max_abs_err": float("inf")}
    return {"max_abs_err": scorers.max_abs_err(got, ref)}
```

## C++ task (`"native": "cpp"`)

```
sol.hpp      the contract: declarations + what they must do
main.cpp     the fixed driver — owns fixtures, models and printing
ref.cpp      the correct implementation
starter.cpp  what the learner starts from (empty body, or the broken one)
```

Grading compiles `main.cpp` with the candidate and with `ref.cpp` using
`clang++ -O2 -std=c++20`, runs both, and diffs the numbers they print. So the
lesson has to be **visible in the output**.

Timing is not reproducible, so a CPU-performance task never measures time. The
driver models what matters — a set-associative LRU cache fed by the addresses
the learner's code touches — and prints counts.

## CUDA task (`"native": "cuda"`)

```
check.py    def grade(srcfile="solve.cu") -> dict
ref.cu      the correct kernel
starter.cu  the starter kernel
```

`check.py` parses the `.cu` with `mlsys.sim.CudaProgram` and executes it
thread-by-thread on `mlsys.sim.GPU`, then gates on correctness **and** on a
hardware behaviour the task is about: `transactions`, `smem_waves`,
`divergences`, `cycles`.

The front end accepts a real subset of CUDA-C: `__global__`, pointer and scalar
parameters, `threadIdx/blockIdx/blockDim/gridDim` with `.x`/`.y`, indexing,
`__shared__` arrays, `__syncthreads()`, `__shfl_up_sync` / `__shfl_down_sync` /
`__shfl_xor_sync`, control flow, and the usual float builtins. It rejects
structs, multidimensional arrays, `break`/`continue`, and the preprocessor.

## Verifying

```bash
tools/verify_task.sh   <id>     # python
tools/verify_native.sh <id>     # cpp / cuda
```

`TASK_OK` is printed only when the reference **passes** every gate and the
shipped starter **fails** at least one. A task that everything passes teaches
nothing, and it is treated as broken.

## Starter vs attempt

Every task ships a **starter** and never ships an attempt:

| track | ships | learner edits |
|---|---|---|
| python | `starter.py` | `solve.py` |
| C++ | `starter.cpp` | `solve.cpp` |
| CUDA | `starter.cu` | `solve.cu` |

The editor seeds `solve.*` from the starter the first time a task is opened, and
`solve.*` is git-ignored. Grading therefore can never overwrite the shipped
starter — which it did while the native tracks used one file for both.
