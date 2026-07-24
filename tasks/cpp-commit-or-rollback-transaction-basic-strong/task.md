## Context

A function provides the **strong exception guarantee** if it either completes fully, or throws and leaves the program state *exactly* as it was before it was called -- no partial mutation. This is usually achieved with a staging buffer or copy-and-swap: copy the original state, mutate the copy, and only commit the copy back with a single non-throwing write once every step has succeeded.

A weaker (and much easier to write by accident) property is the **basic** exception guarantee: nothing leaks or corrupts, but if you throw partway through a sequence of direct writes, whatever was already written stays written.

## Task

`Record` (declared in `sol.hpp`) holds four fields. `transactional_update(rec, ops, numOps, throwAt)` applies `ops[0..numOps)` to `*rec`, one at a time. If `throwAt` equals the (0-based) index of an operation, it must throw `TxnAbort()` instead of applying that operation (and every later one).

The shipped `solve.cpp` applies each op straight into `*rec` as it goes -- only the basic guarantee. Fix it to provide the **strong** guarantee: read the current fields into local copies, apply every op to the copies, and only write the copies back into `*rec` in one non-throwing commit at the very end, once the entire sequence has succeeded. On a throw, `*rec` must come out exactly as it went in.

## Example

```cpp
Op ops[2] = {{Field::Score, 9.5}, {Field::Flags, 7}};
transactional_update(&rec, ops, 2, /*throwAt=*/1);
// 1. copy rec's fields into locals
// 2. apply Score -> 9.5 to the local copy
// 3. op index 1 == throwAt: throw TxnAbort() -- *rec was never touched
```

## What the gate checks

`main.cpp` runs the same four-op sequence with `throwAt` in `{-1, 0, 1, 2, 3}` (`-1` means it never throws), resetting `*rec` to the same starting state each time, and prints whether it threw plus every field of `*rec` afterwards. With the strong guarantee, a throw at index 1, 2, or 3 leaves `*rec` completely unchanged; the basic-guarantee bug leaves whichever fields were processed before the throw already mutated. Your printed numbers are compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`.
