## Context

In multi-threaded C++, the `std::atomic` memory-ordering model defines how compilers and hardware are allowed to reorder memory operations. For two adjacent operations `op1` then `op2` (in program order), the standard's rules boil down to:

1. **Address dependency**: if `op1` and `op2` touch the exact same field, reordering is **forbidden** ($0$) — a single thread must never observe its own operations out of order on one location.
2. **`acquire` barrier on `op1`**: nothing may move *before* an acquire (`memory_order_acquire` or `memory_order_acq_rel`) — **forbidden** ($0$).
3. **`release` barrier on `op2`**: nothing may move *after* a release (`memory_order_release` or `memory_order_acq_rel`) — **forbidden** ($0$).
4. **`seq_cst` on either op**: sequential consistency forbids reordering across it — **forbidden** ($0$).
5. Otherwise — different locations, `op1` not acquire/seq_cst, `op2` not release/seq_cst — reordering is **permitted** ($1$). In particular, a `release` store followed by an `acquire` load on *different* locations still permits that Store-Load reorder.

## Task

`sol.hpp` declares `MemOp` (an operation's `type`, `field_idx`, and `order`) and `ReorderCase` (an `op1`/`op2` pair). Implement:

```cpp
void predict_reordering_legality(const ReorderCase* cases, int n, int* out);
```

Apply the five rules above, in order, to each of the `n` cases and write `1` (permitted) or `0` (forbidden) into `out[i]`.

## Example

`op1 = {Write, field 0, relaxed}`, `op2 = {Read, field 1, relaxed}` → different fields, neither an acquire/release/seq_cst barrier → permitted, `1`. `op1 = {Read, field 0, acquire}`, `op2 = {Write, field 1, relaxed}` → `op1` is an acquire → forbidden, `0`, even though the fields differ.

## What the gate checks

`main.cpp` runs a fixed set of $12$ cases spanning every rule (same-field dependency, `acquire` on `op1`, `release` on `op2`, `seq_cst` on either side, plain relaxed pairs, and the release-store/acquire-load Store-Load-reorder edge case) and prints each of the $12$ verdicts, one per line. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Checking the rules in the wrong priority (e.g. treating same-field pairs as reorderable just because both sides are `relaxed`) flips exactly the cases that rule was supposed to catch.
