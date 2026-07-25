## Context

A data-dependent branch like `if (x[i] > 0) { ... }` doesn't cost the
same everywhere in a kernel — it costs different amounts depending on
*which 32 elements happen to land in the same warp*. If every element
in a warp's slice of `x` falls on the same side of the threshold, the
warp takes exactly one path: no divergence, no serialization. If the
32 elements split — even just one lane disagreeing with the other 31 —
the warp must issue both the `then` and `else` paths, back to back.

Two warps evaluating the exact same predicate function can have wildly
different costs purely because of how their input data happened to be
laid out — a warp of already-sorted, already-clustered data is free;
a warp straddling a decision boundary pays for both branches every
time.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void eval_predicate(const float* x, float* pred_out, int n, float threshold);
```

For each `i < n`: `pred_out[i] = 1.0` if `x[i] > threshold`, else
`0.0`.

(The grader itself groups your per-element results into warps of 32
and reports, per warp, whether all 32 lanes agreed — 1 serialized path
— or split — 2 serialized paths. Your only job is getting the
per-element predicate right; the aggregation is done independently, not
part of what you write.)

## Example

Warp of 32 elements, 30 positive and 2 negative: your kernel should
still evaluate every element correctly and independently (`pred_out[i]
= 1.0` for the 30, `0.0` for the 2) — the fact that this makes the
warp "divergent" is a property of the *data*, not something your
per-element computation needs to know or handle specially.

## What the gate checks

The grader builds a fixed 128-element input across 4 warps — one
entirely positive, one entirely negative, one alternating (mixed), and
one all-positive except a single negative outlier — launches
`eval_predicate`, and aggregates your per-thread output into per-warp
path counts. It compares that against the same aggregation applied to
an independent numpy evaluation of `x > threshold`. It requires

$$
\mathrm{exact\_match} = 1 \iff \text{every one of the 4 per-warp path counts matches the reference}
$$

On this fixture the reference path counts are `[1, 1, 2, 2]` — the two
uniform warps cost one path each, the mixed warp and the
one-outlier warp both cost two, even though the outlier warp is 31/32
identical.
