## Context

Dropout and other per-element GPU randomness need each output element
to be an *independent* draw — element `i`'s coin flip shouldn't tell
you anything about element `i+1`'s. Since GPUs don't have a built-in
per-thread random generator here, kernels build one from a
deterministic hash: mix a seed and a per-element index through some
arithmetic, take the result mod a large number, and treat the
remainder (rescaled to `[0,1)`) as a "random" draw.

The bug this task fixes is a single missing ingredient: if the hash
only depends on `seed` — the same value in every single thread — every
thread computes the exact same hash, and the exact same threshold
decision. Instead of 1000 independent coin flips, you get *one* coin
flip, broadcast to all 1000 output slots. The output isn't garbage —
it's suspiciously, perfectly uniform in the wrong way: every element
identical.

## Task

`solve.cu`'s `dropout_mask` computes, for each `i < n`, a hash from
`seed` alone. **Fix it** by mixing the per-element index `i` into the
hash input — `(seed + i)` instead of `seed` — so each element gets an
independent point in the hash's output space:

```cuda
int h = (seed + i) * 2654435761;
h = h % 2147483647;
float r = h / 2147483647.0;
out[i] = r >= 0.5 ? 1.0 : 0.0;
```

## Example

`seed = 100`. Buggy: element `0` and element `1` both compute
`h = 100 * 2654435761`, the identical value — identical `r`, identical
`out[0]` and `out[1]`, and every other element too. Fixed: element `0`
computes `h` from `100`, element `1` from `101`, element `2` from
`102` — each a different input to the hash, each landing on an
essentially unrelated point in `[0, 2147483647)`.

## What the gate checks

The grader launches `dropout_mask` over `n = 1000` elements with a
fixed seed, and measures how far the output's mean keep-rate is from
the expected `0.5` (for 1000 independent `Bernoulli(0.5)` draws, the
binomial standard deviation is about `1.6%`, so a correct
implementation should land well within a few percent of `0.5`). It
requires

$$
\mathrm{mean\_dev} = |\overline{\mathrm{out}} - 0.5| \le 0.1
$$

The correlated version measures `mean_dev = 0.5` exactly — every one of
the 1000 elements is the identical value (all `0.0`, in this case), the
single shared coin flip repeated 1000 times. Mixing in the element
index drops it to `0.001`.
