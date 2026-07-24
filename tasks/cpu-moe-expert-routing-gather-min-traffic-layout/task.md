## Context

In a Mixture-of-Experts layer, each token is routed to one of `E` expert
weight matrices. Tokens arrive in an arbitrary order that has nothing to
do with which expert they were routed to. Processing them in that
arrival order means the expert weight vector in cache keeps getting
evicted and re-fetched every time consecutive tokens land on different
experts — even though the *set* of experts actually used is small and
each one's weights would happily stay resident if visited together.

## Task

Implement

```cpp
void moe_gather(const double* weights, const int* expert_id, int T, int W, int E, long base, double* out);
```

`T` tokens are routed via `expert_id[0..T)` to one of `E` experts, each
with a `W`-element weight vector. For every token `t`, touch every one
of expert `expert_id[t]`'s `W` simulated addresses (see `sol.hpp` for the
exact address formula and the `touch_byte` cache hook) and write
`out[t] = sum` of that expert's real weight values (from the `weights`
array).

Correctness of `out[]` does not depend on visiting order — but process
tokens **grouped by expert** (all tokens routed to expert 0, then all
routed to expert 1, ...) so each expert's weight vector is fetched once
and reused by every token that shares it.

## Example

With `E=4` experts of `W=64` floats (256 bytes = 4 cache lines) each and
a 256-byte cache (exactly one expert's footprint), grouping pays exactly
4 misses per expert — `16` total for all `T=32` tokens, since a repeat
touch to an already-resident expert's lines is a hit. Processing the 32
tokens in their given round-robin order (`expert_id[t] = t % 4`, so
consecutive tokens are *never* the same expert) evicts and re-fetches the
full 4-line footprint on essentially every single token: `128` misses —
8x more, for the identical `out[]` values.

## What the gate checks

`exact_match`: the driver prints the miss count and the full 32-token
output for one fixed routing. Processing tokens in their original
(non-grouped) order gives the same `out[]` values but a much higher miss
count, so the printed line fails to match even though the "answer" looks
right; an empty starter fails outright.
