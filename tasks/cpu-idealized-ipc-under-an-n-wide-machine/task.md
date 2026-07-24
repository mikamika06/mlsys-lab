## Context

A superscalar CPU can issue several independent instructions in the same
cycle, but it can never issue an instruction before every value it reads is
ready. Two separate limits bound how fast a stream of instructions can
possibly retire, even with a perfect, oracle scheduler:

1. **The critical path.** If instruction $j$ reads a value instruction $i$
   produces, $j$ cannot *start* until $i$ *finishes*:
   $$\text{finish}(i) = \text{start}(i) + \text{latency}(i), \qquad
   \text{start}(j) = \max_{i \in \text{deps}(j)} \text{finish}(i)$$
   The longest chain of such dependencies, $\text{critical\_path} =
   \max_i \text{finish}(i)$, is a hard floor on the number of cycles —
   no amount of parallelism shortens a single dependency chain.

2. **Issue width.** A $W$-wide machine can retire at most $W$ instructions
   per cycle even when every instruction is fully independent, so $n$
   instructions need at least $n / W$ cycles.

Combining both: the idealized number of cycles is whichever bound is larger,
$\text{ideal\_cycles} = \max(\text{critical\_path},\, n / W)$, and the
idealized IPC is $n / \text{ideal\_cycles}$.

## Task

Implement, in `solve.cpp`:

```cpp
double idealized_ipc(const std::vector<int>& latency,
                      const std::vector<std::vector<int>>& deps,
                      int width);
```

`latency[i]` is instruction `i`'s execution latency in cycles; `deps[i]` is
the list of instruction indices that must **finish** before instruction `i`
may **start** (a DAG — no cycles). Compute:

- `critical_path` — the longest finish time over all instructions, using the
  recurrence above (`start(i) = 0` if `deps[i]` is empty).
- `width_bound = n / (double)width`.
- `ideal_cycles = max(critical_path, width_bound)`.
- return `n / ideal_cycles`.

## Example

The driver (`main.cpp`, fixed) runs four hand-built DAGs:

- **A** — 16 independent 1-cycle instructions, `width=4`: no dependencies at
  all, so `critical_path=1` but `width_bound=16/4=4`; IPC is bound by
  width: $16/4 = 4.0$.
- **B** — a pure serial chain of 10 1-cycle instructions, `width=4`: every
  instruction depends on the last, so `critical_path=10` regardless of how
  wide the machine is; IPC is bound by the chain: $10/10 = 1.0$.
- **C** — a mixed DAG of 12 instructions: a 4-long chain of latency-2
  instructions (`0->1->2->3`, critical contribution 8), a 2-long chain of
  latency-3 instructions (`4->5`, contribution 6), and 6 independent
  1-cycle instructions, with `width=4`. `critical_path=8`,
  `width_bound=12/4=3.0`, so `ideal_cycles=8` and $\text{ipc}=12/8=1.5$.
- **D** — the *same* DAG as C but `width=1`. The dependencies don't change,
  so `critical_path` is still 8, but `width_bound=12/1=12.0` now dominates:
  `ideal_cycles=12`, $\text{ipc}=12/12=1.0$.

```
A n=16 width=4 ipc=4.000000
B n=10 width=4 ipc=1.000000
C n=12 width=4 ipc=1.500000
D n=12 width=1 ipc=1.000000
```

The starter always returns `0.0`, which fails every one of the four cases.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires every printed IPC value to be within `1e-6` of the
reference (`main.cpp` + `ref.cpp`) (`max_abs_err <= 1e-6`). Computing
`critical_path` as a plain instruction *count* along the longest chain
(ignoring per-instruction `latency`) gets case B right (all latencies are 1)
but is wrong on C and D, where the two chains have different per-instruction
latencies; forgetting the `max` with `width_bound` (or using it unmodified,
without dividing by `width`) is wrong on A and D, where the width bound is
what actually dominates.
