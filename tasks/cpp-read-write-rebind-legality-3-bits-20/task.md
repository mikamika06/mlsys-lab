## Context

A pointer or reference declaration bundles together three independent
questions:

1. **may_read** — can you dereference/use it to read a value?
2. **may_write_through** — can you assign *through* it, mutating what it
   points/refers to?
3. **may_rebind** — can the declared *name itself* later be made to
   point/refer to a different object?

`const` placement controls the first two: `const T*` means the *pointee*
is const (no write-through), `T* const` means the *pointer variable* is
const (no rebind), and you can combine both. References collapse the third
question entirely: **a reference can never be rebound, ever, regardless of
what it refers to** — there is no syntax in C++ that even attempts it,
`r = other;` always means "assign into whatever `r` refers to", never
"make `r` refer to something else". `void*` collapses the first two: you
cannot dereference a `void*` at all without an explicit cast, so both
reading and writing through it directly are illegal, even though the
pointer variable itself can still be freely reassigned.

## Task

For each of the 20 declarations below (each initialized to refer to some
already-existing object), implement

```cpp
void predict_legality(int out[60]);
```

filling `out[i*3 + 0]` = may_read, `out[i*3 + 1]` = may_write_through,
`out[i*3 + 2]` = may_rebind (each `0` or `1`) for declaration `i`:

```cpp
//  0. int* p                                     10. const char& r
//  1. const int* p                                11. void* p
//  2. int* const p                                12. const void* p
//  3. const int* const p                          13. void* const p
//  4. int& r                                      14. int*& rp
//  5. const int& r                                15. int* const& rp
//  6. double* p                                   16. const int*& rp
//  7. const double* p                             17. const int* const& rp
//  8. double* const p                             18. long* const p
//  9. char& r                                     19. const long* p
```

(`rp` in 14-17 is a reference *to a pointer* — apply the same three
questions to the pointer `rp` names, remembering rule 3 above: since `rp`
itself is a reference, its own may_rebind is always `0`, no matter what
kind of pointer it refers to.)

## Example

`const int* const p` (declaration 3): reading `*p` is fine (`1`), writing
`*p = 1` is illegal because the pointee is `const` (`0`), and reassigning
`p` itself is illegal because the pointer variable is `const` (`0`) →
`{1, 0, 0}`.

## What the gate checks

For every declaration, the driver generates a real, tiny `.cpp` program
that performs exactly the read or write-through operation and asks the
real `clang++ -fsyntax-only -std=c++20` whether it compiles — that
pass/fail is the ground truth for those two bits, never hardcoded. The
may_rebind bit is tested the same way for pointer declarations
(`p = &other;`); for reference declarations it is simply `0`, since no
C++ syntax exists to even attempt rebinding a reference.

The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference across all 60 predicted bits.
