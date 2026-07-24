## Context

Local variables in C++ have **automatic storage duration**: their lifetime
is tied to the lexical scope they were declared in. Leaving a scope —
whether by falling off the end of it, `return`ing, or breaking out of a loop
— destroys every object declared in it, in the **exact reverse order** of
construction. That's true regardless of how many scopes are nested inside
each other; each scope just handles its own objects when it's left.

An exception changes *how* a scope gets left, not the rule itself. When a
`throw` happens, the C++ runtime performs **stack unwinding**: it walks back
through every currently active scope, from the innermost outward, and
destroys every object still alive in each one — again in reverse
construction order — until it finds a matching `catch`. Because scopes were
opened in chronological order and each one destroys its own contents in
reverse chronological order, the *whole* unwind ends up destroying every
live object across every open scope in one single reverse-chronological
sequence: last-constructed-anywhere dies first.

An object that was already destroyed by an ordinary scope exit (`END`)
*before* the throw happened is not part of the unwind — it's gone already.

## Task

Implement

```cpp
long run_trace(const Stmt* stmts, int n, int* out_ids, int* out_count);
```

which replays `stmts[0..n)` in order, maintaining a stack of open scopes
(each holding the ids/bytes of the objects constructed in it, in
construction order):

- `BEGIN` — push a new (empty) scope.
- `END` — pop the current (innermost) scope. This is *ordinary* destruction
  — do not record anything for it.
- `CONSTRUCT` — append `(id, bytes)` to the current (innermost) scope.
- `THROW` — stop here. Walk every open scope **innermost first**, and within
  each scope, **most-recently-constructed first**; write each object's `id`
  into `out_ids` in that order, set `*out_count` to how many there were, and
  return the sum of their `bytes`.

If the trace ends without ever reaching a `THROW`, set `*out_count = 0` and
return `0`.

## Example

```
BEGIN
CONSTRUCT 1 (4 bytes)
BEGIN
CONSTRUCT 2 (1 byte)
END                      <- normal destruction of 2, not part of any unwind
CONSTRUCT 3 (8 bytes)
THROW                    <- unwinds the live objects: 3, then 1
```

`out_ids = [3, 1]`, `*out_count = 2`, return value `8 + 4 = 12`.

## What the gate checks

The driver runs three fixed traces — a throw three scopes deep after one
object was already destroyed normally, a trace that never throws, and a
throw with nothing yet live — using the REAL `sizeof()` of small C++ structs
(including one with real alignment padding) as each object's byte count, not
a hand-computed table. It prints the unwound ids, the count, and the total
for each trace. The grader compiles `solve.cpp` with `clang++ -O2
-std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed id, count, and total matches the reference, in all three traces}
$$

Destroying objects in construction order (instead of reverse), or destroying
an object a normal `END` already removed, both produce a list with the right
*members* but the wrong *order* or *membership* — the gate requires the
exact sequence, not just the right set of ids.
