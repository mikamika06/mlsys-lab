## Context

The **store-buffering** litmus test (a fragment of Dekker's mutual-exclusion
algorithm) is the canonical example that separates *sequential consistency* from
weaker memory models. Two shared atomics start at $x = y = 0$:

$$
\text{Thread A:}\quad x \gets 1;\ \ r_1 \gets y
\qquad\qquad
\text{Thread B:}\quad y \gets 1;\ \ r_2 \gets x
$$

Each thread writes one location, then reads the *other*. The question is which
final pairs $(r_1, r_2)$ a conforming C++ implementation may produce.

Under **`memory_order_seq_cst`** every operation joins one global total order, so
the executions are exactly the interleavings of the four operations. To read
$r_1 = 0$, thread A's load of $y$ must precede B's store of $y$; to read
$r_2 = 0$, B's load of $x$ must precede A's store of $x$. Combined with
program order ($x\gets1$ before $r_1\gets y$, and $y\gets1$ before $r_2\gets x$)
this forms a cycle, so the outcome $(0,0)$ is **forbidden**.

Under **relaxed** ordering each thread may keep its store in a private store
buffer (x86-TSO) and read the other location *before* that store — or the other
thread's store — becomes globally visible. Now both loads can read the stale $0$,
so $(0,0)$ becomes **observable**.

Encode an outcome $(r_1, r_2)$ as the 2-bit index $(r_1 \ll 1)\,|\,r_2$:

| index | outcome |
|-------|---------|
| 0     | (0,0)   |
| 1     | (0,1)   |
| 2     | (1,0)   |
| 3     | (1,1)   |

## Task

Implement the contract in `sol.hpp`:

- `int allowed_outcomes(bool store_buffering)` — return a 4-bit mask; bit `i` is
  set iff outcome `i` is observable. When `store_buffering == false` model
  seq_cst (enumerate the interleavings). When `store_buffering == true` model the
  store-buffering relaxation (a store may sit buffered while the thread reads on).

- `void sc_outcome_histogram(int counts[4])` — over all six sequentially-consistent
  interleavings of the four operations (respecting per-thread program order), set
  `counts[i]` to the number of interleavings that produce outcome `i`.

The recommended approach is an explicit operational search over program states
rather than hard-coding the answer.

## Example

The fixed driver builds no input; it just calls your functions and prints:

```
14
15
0 1 1 4
```

Line 1 is the seq_cst mask `0b1110` (outcomes 1, 2, 3 — `(0,0)` forbidden).
Line 2 is the relaxed mask `0b1111` (all four, `(0,0)` now allowed).
Line 3 is the interleaving histogram: `(0,0)` never occurs, `(0,1)` and `(1,0)`
occur once each, and `(1,1)` occurs in 4 of the 6 interleavings.

## What the gate checks

The grader compiles `main.cpp` + your source with
`clang++ -O2 -std=c++20`, runs it, and compares the printed output to the
reference character-for-character (`exact_match == 1.0`). All three lines — both
masks and all four histogram counts — must match exactly.
