## Context

A `small_vector` keeps its first few elements in an inline buffer embedded in the
object itself, avoiding a heap allocation for the common "just a handful of
elements" case, and only "spills" to the heap once it grows past that inline
capacity $N$. Standard library-quality containers implement this by managing
element lifetimes *by hand*: the inline buffer is raw, correctly aligned storage
(`alignas(T) unsigned char[sizeof(T) * N]`), and elements are created with
**placement new** and torn down with **explicit destructor calls**. Getting this
right means every element is constructed exactly once and destroyed exactly once —
no leaks, no double-free, no double-construct — even across a reallocation.

This task uses a fixed element type `Tracked` that instruments its own lifetime:

$$\texttt{alive} \;=\; \texttt{ctor\_calls} - \texttt{dtor\_calls}$$

If your container manages lifetimes correctly, `alive` returns to $0$ once the
container is destroyed and `ctor_calls == dtor_calls` throughout.

## Task

Implement the member functions of `SmallVector` (declared in `sol.hpp`) in
`solve.cpp`. The storage layout is fixed for you: an inline buffer `inbuf` for
`CAP = 4` elements, a `data` pointer, a size `sz`, and a capacity `cap`.

- `SmallVector()` — start empty with `data` pointing at the inline buffer.
- `push_back(long v)` — append `Tracked{v}` via placement new. When `sz == cap`,
  first grow (double the capacity) onto the heap by copy-constructing the
  existing elements into the new block, destroying the old ones, and freeing any
  old heap memory.
- `sum()` — sum of every element's `value`.
- `size()` — number of live elements.
- `spilled()` — `true` iff storage currently lives on the heap.

The inline buffer must be used while `size() <= CAP`; the first push past `CAP`
must move the elements onto the heap.

## Example

```
push 3,1,4,1  -> 4 elements, still inline (spilled = false)
push 5        -> grows to heap capacity 8 (spilled = true)
push 9,2,6    -> 8 elements total, sum = 31
destroy       -> alive back to 0, ctor_calls == dtor_calls
```

## What the gate checks

`main.cpp` pushes the deterministic sequence `3 1 4 1 5 9 2 6` (which overflows
the inline buffer and forces a heap spill), reads back `size`, `sum`, and
`spilled`, then destroys the container and inspects the lifetime counters. It
prints `size`, `sum`, `spilled`, `alive_during`, `alive_after`, `ctor_eq_dtor`,
and `leaked`. The grader compiles `main.cpp` + your `solve.cpp` with
`clang++ -O2 -std=c++20` and requires the printed output to match the reference
exactly ($\mathrm{exact\_match} = 1.0$): correct values **and** correct lifetime
bookkeeping (`alive_after = 0`, `leaked = 0`).
