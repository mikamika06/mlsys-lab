## Context

Real CPUs have several levels of cache between the core and DRAM — L1
(smallest, fastest), L2 (bigger, slower), L3 (biggest, slowest) — and most
real hierarchies are **inclusive**: every line in L1 must also be present in
L2, and every line in L2 must also be present in L3. The consequence: when a
line is evicted from L2 (to make room for something else), it must also be
invalidated from L1, even if L1 itself has plenty of free space — because
inclusion says L1's contents must always be a subset of L2's. The same
cascade applies when L3 evicts a line: it must be invalidated from both L2
and L1.

Real hardware cache timing isn't reproducible across machines, so this task
grades against a small deterministic model instead (declared in `sol.hpp`,
implemented in `main.cpp`): a 3-level inclusive hierarchy with a 512-byte
L1 (direct-mapped), a 4096-byte L2 (4-way), and a 16384-byte L3 (8-way),
all with 64-byte lines. You don't need to reimplement the model — you drive
it by calling `touch(addr)`, and the model does the rest, including
inclusion.

## Task

Implement

```cpp
void access_pattern(int N);
```

which touches every element of an `N x N` matrix of 4-byte elements, stored
**row-major** (element `(row, col)` lives at byte address
`(row * N + col) * 4`), by calling `touch()` on every element's address
**exactly once**. You choose the loop order.

## Example

A 64-byte cache line holds 16 consecutive `float`s. If you iterate with
`col` as the innermost loop variable, 16 consecutive touches land in the
same line before moving to the next one — each line is only fetched once.
If you iterate with `row` innermost instead, consecutive touches jump
`N * 4` bytes apart every time (a full row's worth), landing in a
*different* line on almost every single touch — the classic
transpose-style access pattern, and a real, common source of cache misses
in code that "obviously" just walks a matrix.

## What the gate checks

The driver runs `access_pattern(32)` (a 4096-byte matrix — fits inside L2
and L3, but not L1) and prints the resulting `L1_misses`, `L2_misses`,
`L3_misses`. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{all three printed miss counts match the reference}
$$

At this working-set size, `L2_misses` and `L3_misses` sit at their
compulsory-miss floor (64 — the number of distinct 64-byte lines, since the
whole matrix fits in both) **regardless of loop order** — but `L1_misses`
does not: the cache-friendly (column-innermost) order measures 64 as well,
while the pathological (row-innermost) order measures 1024 — every single
touch misses L1, since consecutive addresses never share a line. Getting
the loop order backwards is invisible in the *result* of a real program (no
values are computed here, only addresses are touched), but not invisible in
the *miss count* — which is exactly what a profiler would show you.
