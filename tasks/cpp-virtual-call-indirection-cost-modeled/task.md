## Context

C++ dynamic polymorphism is implemented with a **vtable**. Every object of a
polymorphic class stores a hidden **vptr** (vtable pointer) at offset 0. Calling
a virtual method is therefore not a direct jump: the CPU must chase pointers.

Model each virtual call as costing two *dependent* memory reads — "indirect
loads":

1. the **vptr load** — read the object's vtable pointer, and
2. the **vtable-slot load** — read the target function pointer out of the vtable.

So a naive virtual call costs $2$ indirect loads. A call the compiler can
**devirtualize** (resolve statically or inline) costs $0$: the address is known
at compile time, so no pointer chasing happens.

Real compilers also keep recently loaded values in registers. If two consecutive
calls hit the same receiver, the vptr is already in a register; if they hit the
same receiver *and* the same method, the function pointer is too. This is the
gap between the textbook $2n$ figure and what actually executes.

## Task

Implement the three functions declared in `sol.hpp`. A call trace of length `n`
is given as two parallel arrays: `obj_id[i]` (receiver identity at call site `i`)
and `slot[i]` (vtable slot / method index at call site `i`). Return the total
modeled indirect loads under each policy:

- `naive_virtual_loads` — $2$ loads for every call: $2n$.
- `cached_virtual_loads` — with register reuse across *consecutive* calls:
  - skip the vptr load when `obj_id[i] == obj_id[i-1]`;
  - skip the vtable-slot load when `obj_id[i] == obj_id[i-1]` **and**
    `slot[i] == slot[i-1]`;
  - the first call (`i == 0`) pays both loads.
- `devirtualized_loads` — every call resolved at compile time: $0$.

The fixed driver in `main.cpp` runs a deterministic 20-call trace and prints
`naive`, `cached`, `devirt`, and `saved = naive - cached`.

## Example

```
trace:  obj = [7, 7, 7, 4, 4]
        slot = [0, 0, 1, 0, 0]

naive  = 2 * 5                    = 10
cached:
  i=0  7,0  first call            -> vptr + slot = 2
  i=1  7,0  same obj & slot       -> 0
  i=2  7,1  same obj, new slot    -> slot only  = 1
  i=3  4,0  new obj               -> vptr + slot = 2
  i=4  4,0  same obj & slot       -> 0
                                     cached      = 5
saved  = 10 - 5                   = 5
```

## What the gate checks

The grader compiles `main.cpp` together with your source using
`clang++ -O2 -std=c++20`, runs it, and compares the printed lines against the
reference solution's output exactly ($\mathrm{exact\_match} = 1.0$). Every
number — `naive`, `cached`, `devirt`, `saved` — must match.
