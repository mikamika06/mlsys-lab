## Context

A function's *static* instruction count — `len(list(dis.get_instructions(f)))`
— counts the bytecode operations CPython compiled the function body into. It
doesn't change when the function is called (CPython 3.11's adaptive
interpreter specializes individual opcodes in place after enough calls, but
it doesn't add or remove instructions, so the count stays stable), which
makes it a clean, deterministic proxy for "how much control-flow scaffolding
did this implementation need." Chains of `if`/`else` compile to a
`LOAD`/`COMPARE_OP`/jump/`STORE` for every branch, whereas a single
expression built from builtins compiles to a handful of `LOAD`/`CALL`
instructions.

Consider clamping a value into `[lo, hi]`. Written out branch by branch:

```python
def clamp_verbose(x, lo, hi):
    result = None
    if x < lo:
        result = lo
    else:
        if x > hi:
            result = hi
        else:
            result = x
    return result
```

This is correct, but it compiles to over twenty instructions. The same
behavior is exactly

$$
\operatorname{clamp}(x, lo, hi) = \max\bigl(lo,\ \min(x, hi)\bigr),
$$

which needs far fewer.

## Task

Implement `clamp(x, lo, hi)`:

```python
def clamp(x: float, lo: float, hi: float) -> float:
    ...
```

Given `lo <= hi`, return `x` bounded into `[lo, hi]`: `lo` if `x < lo`,
`hi` if `x > hi`, otherwise `x`. Your implementation's compiled bytecode,
measured as `len(list(dis.get_instructions(clamp)))`, must be small — favor
a direct expression over an explicit branch tree.

## Example

```python
clamp(-3, 0, 10)   # -> 0
clamp(15, 0, 10)   # -> 10
clamp(4, 0, 10)    # -> 4
clamp(5, 5, 5)     # -> 5
```

## What the gate checks

The gate first checks correctness: it evaluates `clamp` on a batch of
`(x, lo, hi)` triples — random floats and integers plus boundary cases
(`x` exactly at `lo`, exactly at `hi`, `lo == hi`, negative ranges) — against
the closed-form reference `max(lo, min(x, hi))`, computed directly, not
hardcoded per case. Every case must match exactly.

Second, it inspects your function object with the real `dis` module:
`len(list(dis.get_instructions(sol.clamp)))` must be at most `14`. The
branch-tree implementation shown above compiles to `21` instructions and
would fail this budget even though its output is perfectly correct — the
gate is specifically testing that you reach for a compact expression
instead of restating the branches.
