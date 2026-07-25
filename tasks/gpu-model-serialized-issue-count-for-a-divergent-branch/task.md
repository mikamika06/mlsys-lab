## Context

A warp executes one instruction stream for all 32 lanes at once. When
an `if/else` sends some lanes down the `then` branch and others down
`else`, the hardware can't run two different instruction streams
simultaneously — it runs **both** paths, back to back, masking off
whichever lanes aren't "supposed" to be active on each pass. A lane
taking the `then` branch sits idle (predicated off) while the `else`
path issues, and vice versa. The total issued instruction count for
the warp is the sum of both paths' lengths — not the length of whichever
path any individual lane actually needed.

Only when every lane in the warp agrees — all 32 take `then`, or all 32
take `else` — does the warp skip the path it never uses. That's the one
case where branching is free.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void divergent_issue_count(int then_instrs, int else_instrs,
                                       int lanes_taking_then, float* out);
```

Given the instruction counts of each branch and how many of the warp's
32 lanes take `then`:

- If `lanes_taking_then == 0`: `issues = else_instrs` (uniformly else).
- If `lanes_taking_then == 32`: `issues = then_instrs` (uniformly then).
- Otherwise (divergent): `issues = then_instrs + else_instrs` — both
  paths serialize.

Write `issues` to `out[0]`. Also compute the **penalty factor** against
the more expensive of the two paths alone —
`penalty = issues / max(then_instrs, else_instrs)` — and write it to
`out[1]`.

Note: `then_instrs`/`else_instrs`/`lanes_taking_then` are `int`
parameters. Computing `issues / max(...)` needs true (floating-point)
division — adding `0.0` to an int parameter (`then_instrs + 0.0`)
before using it forces that.

## Example

`then_instrs=10, else_instrs=6, lanes_taking_then=12`: divergent (`12`
is neither `0` nor `32`), so `issues = 10 + 6 = 16`. The more expensive
single path alone would have been `10`, so `penalty = 16/10 = 1.6` — a
60% overhead purely from the warp disagreeing on which way to branch.

## What the gate checks

The grader launches `divergent_issue_count` for 6 fixed scenarios
(covering both-uniform and various divergent splits) and compares both
outputs against an independently computed oracle. It requires

$$
\mathrm{exact\_match} = 1 \iff \text{both outputs match the oracle on every one of the 6 scenarios}
$$

On `(then=8, else=8, lanes\_taking\_then=1)` — a *maximally* divergent,
perfectly balanced branch — the penalty comes out to exactly `2.0`: the
warp pays for both paths in full, twice the cost of either one alone,
even though only one lane out of 32 needed the `then` side at all.
