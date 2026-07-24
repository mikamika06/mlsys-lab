## Context

Every C++ object's lifetime is a sequence of events — a constructor runs
once, maybe a copy or move constructs another object from it, and
eventually a destructor ends it. Which events fire, and in what order,
follows directly from the code: how many local variables are declared, what
each is initialized from (a fresh value? another object, by copy? another
object, by `std::move`?), and where scopes close (which destroys everything
declared inside them, in **reverse** declaration order).

`Probe` (declared in `sol.hpp`) logs a character every time one of its
special members runs: `C` (construct), `Y` (copy construct), `M` (move
construct), `D` (destruct). Running a piece of code that uses `Probe`
produces a string you can read directly off the log — turning "what events
does this code cause" from something you reason about silently into
something you can literally print and check.

## Task

Write the **body** of

```cpp
void reproduce_sequence();
```

using local `Probe` variables, copies, moves, and nested scopes (`{ ... }`),
so that running it logs **exactly** this sequence, in order:

```
C Y M D D D
```

That is: one `Probe` constructed; copy-constructed into a second `Probe`;
that second `Probe` move-constructed into a third; then all three
destroyed, with the copy and its move-target destroyed **before** the
first `Probe` — which means they must go out of scope first.

## Example

```cpp
void reproduce_sequence() {
    Probe p1(1);              // C
    {
        Probe p2(p1);          // Y
        Probe p3(std::move(p2)); // M
    }                           // p3, then p2 destroyed: D D
}                                // p1 destroyed: D
```

## What the gate checks

The driver calls `reproduce_sequence()` once, then prints the accumulated
log. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it,
and requires

$$
\mathrm{exact\_match} = 1 \iff \text{the printed log is exactly } \texttt{CYMDDD}
$$

Getting the *count* of each event type right (one `C`, one `Y`, one `M`,
three `D`s) but declaring `p3` in the *same* scope as `p1` instead of a
nested one changes the destruction order to `D D D` with `p1` destroyed
last-but-one instead of last — a different string, so the gate still
catches it even though every event fired the right number of times.
