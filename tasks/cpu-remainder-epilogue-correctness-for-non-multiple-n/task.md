## Context

Unrolling a loop by a factor `U` (processing `U` elements per iteration
to cut loop-overhead and expose more independent work) only evenly
covers `n` when `n` is a multiple of `U`. Real problem sizes usually
are not, so every unrolled loop needs a scalar **epilogue** that handles
the `n % U` leftover elements one at a time. Forgetting it is one of the
most common real-world vectorization bugs — it silently produces the
right answer whenever a test happens to use a "nice" `n`, and silently
corrupts the tail whenever it doesn't.

## Task

Fix

```cpp
void scale_unrolled(const float* in, int n, float s, float* out);
```

It must compute `out[i] = s * in[i]` for every `i` in `[0, n)`, using a
4-way unrolled main loop for the largest multiple of 4 `<= n`, **plus** a
scalar loop that handles the remaining `n % 4` elements.

The shipped version is missing that scalar epilogue: it stops after the
last full group of 4 and never touches the leftover elements. Add the
epilogue loop.

## Example

For `n = 13`, the unrolled main loop covers indices `0..11` (3 groups of
4); index `12` is the one leftover element (`13 % 4 = 1`) that only the
epilogue writes.

## What the gate checks

`exact_match`: the driver prints all 13 output elements for a fixed
input where `n` is not a multiple of 4. The missing-epilogue version
gets indices `0..11` exactly right but leaves index `12` at the driver's
`-999.00` sentinel, so the printed line fails to match at that one
position even though everything else is correct.
