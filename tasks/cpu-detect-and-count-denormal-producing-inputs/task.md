## Context

IEEE-754 single precision reserves the smallest exponent field for
**subnormal** (denormal) numbers: values whose magnitude is below the
smallest positive **normal** float,

$$\mathrm{FLT\_MIN} \approx 1.1754944 \times 10^{-38},$$

but which are not zero. Subnormals trade precision (they have progressively
fewer significant bits the smaller they get) for the ability to represent
values that would otherwise underflow straight to zero — "gradual
underflow." They matter for performance too: on most x86/ARM FPUs,
arithmetic that touches a subnormal operand or produces a subnormal result
runs on a slow microcoded path, sometimes 10-100x slower than normal
floats. Detecting that a data buffer contains denormal-range values is the
first step before deciding whether to flush them to zero (DAZ/FTZ) for
speed.

A value $x$ is subnormal iff $0 < |x| < \mathrm{FLT\_MIN}$. Note the
strict `<`: $x = \pm\mathrm{FLT\_MIN}$ itself is the smallest **normal**
float, not subnormal. Zero, infinities, and NaN are their own separate
classes and must not be counted either.

## Task

Implement

```cpp
int count_denormals(const float* arr, int n);
```

Return the count of `i` in `[0, n)` for which `arr[i]` is subnormal:
nonzero, finite, and `fabs(arr[i]) < FLT_MIN`.

## Example

Among `{1.0f, 0.0f, 1e-40f, FLT_MIN, FLT_MIN * 0.5f, -1e-42f, INFINITY}`:
only `1e-40f`, `FLT_MIN * 0.5f`, and `-1e-42f` are subnormal — `0.0f` is
zero, `FLT_MIN` is exactly the normal/subnormal boundary (normal side),
and `INFINITY` isn't finite. The count is `3`.

## What the gate checks

`exact_match`: the driver runs 20 fixed floats — ordinary normals, `+0`/`-0`,
four subnormals spread across the exponent range, a subnormal pair sitting
exactly at `FLT_MIN * 0.5` (the tightest boundary case), `+-FLT_MIN` itself
(normal, must NOT be counted), `+-infinity`, and `NaN` — and prints the
returned count. Forgetting the `x != 0` check overcounts by including the
two zeros; using a loose threshold like `1e-30f` instead of the true
`FLT_MIN` boundary miscounts the `FLT_MIN * 1.5f` / `FLT_MIN * 0.5f` pair;
either mistake changes the printed number and fails the match.
