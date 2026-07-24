## Context

A **single-producer / single-consumer (SPSC) ring buffer** is the workhorse
lock-free queue: one thread pushes, one thread pops, and they coordinate through
a fixed-size circular array plus two indices — a producer index and a consumer
index. In a real concurrent implementation the producer publishes its index with
a *release* store and the consumer reads it with an *acquire* load (and vice
versa), which is exactly enough ordering to guarantee two invariants:

- **No loss** — every element the producer accepts is eventually delivered.
- **No duplication** — every accepted element is delivered *exactly once*, in
  FIFO order.

The whole design hinges on the index arithmetic: distinguishing *full* from
*empty*, and wrapping the physical slot with $\text{slot} = \text{index} \bmod
C$ for capacity $C$. Get the boundary wrong and you either overwrite live data
(loss), hand out the same slot twice (duplication), or deadlock a full/empty
buffer.

This task isolates that arithmetic. The buffer is exercised **single-threaded and
deterministically**, so correctness — not thread timing — is what is graded.

## Task

Implement the three methods of `RingBuffer` in `solve.cpp`:

```cpp
bool   try_push(int32_t v);   // false if full (no overwrite); else store, return true
bool   try_pop(int32_t* out); // false if empty; else copy oldest element, return true
size_t size() const;          // number of buffered elements, 0 .. capacity()
```

Storage is provided in `sol.hpp`: a backing array `buf_` of exactly `cap_`
slots, plus two indices `head_` and `tail_` whose meaning you choose. Your
implementation must:

- Reject `try_push` when the buffer already holds `capacity()` elements, leaving
  the buffer unchanged.
- Reject `try_pop` when the buffer is empty, leaving `*out` unchanged.
- Preserve **FIFO order**: `try_pop` returns the oldest element still buffered.
- Wrap around the circular storage correctly so no accepted element is ever lost
  or duplicated.

You do not write any threading code — just the index logic that makes the
sequence semantics correct.

## Example

With capacity 4, pushing `100,101,102,103,104,105` accepts the first four and
rejects the last two (buffer full), so `size()` is 4. A single `try_pop` returns
`100` and frees one slot, after which a further `try_push` succeeds again.

Streaming the values `0,1,2,...,39` through a capacity-4 buffer while
interleaving pushes and pops must let the consumer read back exactly
`0,1,2,...,39` — same values, same order, none missing, none repeated.

## What the gate checks

The fixed driver `main.cpp` builds deterministic inputs, drives your buffer
through a full/empty probe and a 40-element interleaved stream, and prints the
resulting counts and the full received sequence. The grader compiles
`main.cpp` + your `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
compares the printed output to the reference implementation.

Gate: `exact_match == 1.0` — every printed number (accepted-push count, buffer
size, first popped value, the received length, the in-order flag, the checksum,
and the entire delivered sequence) must match the reference exactly. Any loss,
duplication, reordering, or off-by-one in full/empty detection changes the
output and fails the gate.
