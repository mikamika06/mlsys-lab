## Context

In C++, dynamic polymorphism (virtual dispatch) is implemented under the hood using a **vtable**. For any class containing at least one virtual function, the compiler inserts a hidden pointer (the `vptr`) as the very first word of the object's memory layout. `vptr` points at the class's vtable, an array of function pointers — one per virtual member, in declaration order.

`sol.hpp` defines the hierarchy below with real, ordinary C++ `virtual` functions (this is exactly what the compiler turns into vptr + vtable for you):

```cpp
struct Base {
    virtual double compute(int x) const = 0;
    virtual ~Base() = default;
};

struct Derived1 : Base {
    int factor;
    double weight;
    double compute(int x) const override { return (x * factor) + weight; }
};

struct Derived2 : Base {
    long offset;
    double compute(int x) const override { return x + offset; }
};
```

Because `compute` is declared before the destructor in `Base`, it occupies vtable slot $0$ in every derived class.

## Task

Implement `double manual_dispatch(const Base* obj, int x)` so it produces exactly what `obj->compute(x)` would — **without writing `obj->compute(x)`, `dynamic_cast`, or `typeid`**. Instead:

1. Read the object's `vptr`: the first machine word at `obj`'s address.
2. Index into the vtable it points to and pull out slot $0$ — the raw function pointer for `compute`.
3. Cast that pointer to the right C++ function-pointer type and call it directly, passing `obj` as the implicit `this` argument.

This works because on this platform's Itanium-based C++ ABI, a non-static member function compiled for a class with no multiple/virtual inheritance is just an ordinary function that takes `this` as its first argument using the platform's normal calling convention — the same convention `reinterpret_cast<double(*)(const Base*, int)>` gives you.

## Example

For `Derived1 d1(3, 1.5)` and `x = 5`: `manual_dispatch(&d1, 5)` must return the same `12.5` that `d1.compute(5)` (via `Base::compute`) returns, obtained purely by chasing raw pointers — no knowledge of `Derived1`'s concrete type is used inside `manual_dispatch`.

## What the gate checks

`main.cpp` builds a fixed array of `Derived1`/`Derived2` objects and calls `manual_dispatch` on each through a `const Base*`, printing every result. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's stdout. A wrong vtable slot, a wrong function-pointer signature, or forgetting to pass `this` all produce numbers that silently diverge from the true virtual-call results rather than crashing — which is exactly why hand-rolled vtable code is dangerous, and exactly what this gate catches.
