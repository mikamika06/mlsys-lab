## Context

`std::shared_ptr<T>` needs two things on the heap: the object `T` itself,
and a **control block** (the strong/weak reference counts and, for
`shared_ptr(new T)`, a type-erased deleter). How you construct the
`shared_ptr` determines whether those two things share one allocation or
need two:

- **`std::shared_ptr<T>(new T(...))`** — `new T(...)` is a complete
  expression that allocates `T` *before* `shared_ptr`'s constructor ever
  runs. The constructor then has no choice but to allocate a *separate*
  control block to go with the pointer it was handed. **Two allocations.**
- **`std::make_shared<T>(...)`** — allocates a single block sized for the
  control data *and* a `T`, laid out together, and constructs `T` in place
  inside it. **One allocation.** This is also why `make_shared` is
  preferred by default: fewer allocator round-trips, and better locality
  between the object and its control block.

## Task

Implement

```cpp
std::shared_ptr<Payload> make_payload(bool use_make_shared, int a, double b, char c);
```

- If `use_make_shared` is `true`, construct via `std::make_shared<Payload>`.
- If `false`, construct via `std::shared_ptr<Payload>(new Payload{...})`.
- Either way, the resulting `Payload` must hold `a`, `b`, `c`.

The driver measures this for real: it overrides the **global**
`operator new`/`operator delete` to count every heap allocation (and its
size), which sees every allocation made *inside* `std::make_shared` and
`std::shared_ptr`'s own internals too — you don't count or predict
anything, the numbers come from actually running your code.

## Example

```cpp
make_payload(true,  7, 3.5, 'A');   // std::make_shared        -> 1 allocation
make_payload(false, 7, 3.5, 'A');   // std::shared_ptr<T>(new)  -> 2 allocations
```

## What the gate checks

The driver calls `make_payload` both ways, printing the measured allocation
count, total bytes allocated, the recovered field values, and `use_count()`
for each. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed count, byte total, and field value matches the reference}
$$

Using `make_shared` for both branches (or `new` for both) gets the field
values right but the allocation count wrong on one of the two lines — the
gate is on the real, measured allocator behavior, not just on whether the
`Payload` ends up holding the right numbers.
