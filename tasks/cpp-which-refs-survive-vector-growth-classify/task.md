# Which references survive vector growth (classify)

## Context

`std::vector` stores its elements in a single contiguous block. When that block
has to move -- a **reallocation** -- every previously taken pointer, reference,
and iterator into the vector becomes dangling. The standard pins down exactly
when this happens:

- `push_back` reallocates **iff** $\text{size} = \text{capacity}$ before the call.
- `reserve(k)` reallocates **iff** $k > \text{capacity}$ (otherwise it is a no-op).
- `insert` reallocates **iff** the resulting size would exceed the current capacity.

Even without a reallocation a reference can still be invalidated. For an
insertion or erasure at position $p$, references to elements at index $\ge p$ are
invalidated (those elements shift), while references to elements strictly before
$p$ remain valid. `pop_back` erases the last element; `clear` erases them all.

## Task

Implement `ref_survives` in `solve.cpp` (declared in `sol.hpp`):

```cpp
bool ref_survives(int n0, int cap0, int refIdx, const std::vector<Op>& ops);
```

A `std::vector<long>` is created with size `n0` and capacity **exactly** `cap0`
($0 \le \text{refIdx} < n0 \le \text{cap0}$). A reference is taken to the element
at index `refIdx`. The mutations in `ops` (`RESERVE`, `PUSH_BACK`, `POP_BACK`,
`INSERT`, `CLEAR` -- see `sol.hpp`) are then applied in order. Return `true` iff
the reference to that **original** element is still valid afterward, else `false`.

The driver `main.cpp` runs 12 fixed scenarios and prints one bit per scenario
followed by their popcount.

## Example

Start with size 4, capacity 8, and take a reference to index 1.

- `ops = [PUSH_BACK]`: size $4 < 8$, no reallocation, the append does not touch
  index 1 -> reference **survives** -> `1`.

Now start with size 4, capacity 4, reference at index 1.

- `ops = [PUSH_BACK]`: size $4 = 4$, `push_back` reallocates -> every reference
  dies -> `0`.

## What the gate checks

`main.cpp` is compiled together with your `solve.cpp` using
`clang++ -O2 -std=c++20` and run. The 12 printed bits (plus popcount) must match
the reference implementation exactly (`exact_match == 1.0`). The starter always
returns `false`, so it prints all zeros and fails the gate.
