## Context

**Placement new** constructs an object inside memory you already own,
instead of asking the allocator for a fresh block:

```cpp
T* p = ::new (buf) T(args...);   // construct a T AT buf, no allocation
```

Because no allocation happened, `delete p` is illegal here — `delete` pairs
a destructor call with a call to `::operator delete`, and `buf` was never
handed out by `::operator new`. Ending the object's lifetime instead
requires an **explicit destructor call**:

```cpp
p->~T();   // destroy the object; buf itself is untouched, still yours
```

This pattern is the building block behind arena allocators, object pools,
and `std::optional`/`std::variant`-style in-place storage: the memory's
lifetime and the object's lifetime are managed separately.

## Task

Implement

```cpp
void placement_lifecycle(void* buf, int a, double b, int* out_a, double* out_b);
```

1. Construct exactly one `Probe(a, b)` into `buf` with placement new.
2. Copy the constructed object's `a` and `b` fields into `*out_a` /
   `*out_b`.
3. End its lifetime with an explicit destructor call, `p->~Probe()`.

`buf` is already sized and aligned correctly for a `Probe` — that's the
driver's job, not yours.

## Example

```cpp
alignas(Probe) unsigned char buf[sizeof(Probe)];
int a; double b;
placement_lifecycle(buf, 7, 2.5, &a, &b);
// exactly one Probe constructed, exactly one destroyed; a == 7, b == 2.5
```

## What the gate checks

`Probe`'s constructor and destructor are instrumented to count how many
times each actually runs. The driver prints those counts, the recovered
field values, and the real `sizeof(Probe)`/`alignof(Probe)`. The grader
compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{ctor\_count} = \text{dtor\_count} = 1 \text{ and every printed value matches the reference}
$$

Forgetting the explicit destructor call passes the field-value check but
leaves `dtor_count = 0` — a leaked lifetime the gate catches even though the
data itself looks fine.
