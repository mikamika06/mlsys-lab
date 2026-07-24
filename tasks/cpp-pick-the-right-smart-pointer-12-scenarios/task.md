## Context

C++ offers three main smart pointer types for automatic resource
management: `std::unique_ptr` (exclusive ownership, zero-overhead),
`std::shared_ptr` (shared ownership via reference counting), and
`std::weak_ptr` (a non-owning observer used to break reference cycles).
Raw pointers (`T*`) remain the right choice for non-owning references or
interop with legacy C APIs. Picking the wrong type causes double-frees,
memory leaks, or dangling references.

## Task

Implement, in `solve.cpp`,

```cpp
std::vector<std::string> smart_pointer_selection();
```

Return 12 strings, each exactly one of `"unique"`, `"shared"`, `"weak"`,
or `"raw"` — the single most appropriate pointer type for each scenario,
in this order:

```
 1. A resource exclusively owned by a single class, automatically
    freed when the class instance goes out of scope.
 2. A resource shared among multiple subsystems; any subsystem may
    outlive others, but the resource must persist until the last
    subsystem is destroyed.
 3. A cache storing recently used objects, where the cache itself owns
    the objects (external code only observes them without affecting
    their lifetime).
 4. A factory function creates an object and returns it to the caller,
    who takes over ownership.
 5. A non-owning reference to an object whose lifetime is guaranteed to
    exceed the reference.
 6. A graph where parent nodes own child nodes, but child nodes need
    back-pointers to parents without creating reference cycles.
 7. A resource passed to a legacy C API expecting a raw pointer; the API
    does not take ownership and will not free it.
 8. A shared ownership scenario where one thread creates the resource,
    and multiple threads read it; the resource is destroyed only after
    all threads finish.
 9. A polymorphic base class destructor that must be virtual; the
    derived object is allocated with `new` and managed by a smart
    pointer.
10. A non-owning pointer used in a performance-critical inner loop where
    atomic reference-counting overhead is unacceptable.
11. A factory returning a pimpl handle; the handle is copyable and
    shares ownership of the implementation.
12. A non-owning pointer stored in an object that is itself managed by a
    std::shared_ptr, forming a potential cycle.
```

## Example

Scenario 1 (exclusive ownership, freed on scope exit) is the textbook
`unique_ptr` case: `"unique"`. Scenario 6 (owning-graph back-pointer that
must not create a cycle) is the textbook reason `weak_ptr` exists:
`"weak"`.

## What the gate checks

The fixed driver (`main.cpp`) calls `smart_pointer_selection()` and
prints the 12 labels, one per line. The gate is an exact string match
(`exact_match == 1.0`) against the reference's printed output — every one
of the 12 scenarios must get the single best-fitting answer.
