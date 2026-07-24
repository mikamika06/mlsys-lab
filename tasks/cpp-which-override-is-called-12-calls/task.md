## Context

C++ member calls resolve in two different ways. A **virtual** call is dispatched on
the object's **dynamic type** (the most-derived type of the object the pointer or
reference actually refers to), typically through the vtable. A **non-virtual** call
is bound at compile time to the **static type** of the expression — the declared
type of the object, pointer, or reference — regardless of what it points at.

Two more wrinkles matter here:

- A call on a **value object** (e.g. `b.who()`) always uses that object's own static
  type; there is no polymorphism to a different type.
- **Object slicing**: copying a `Derived` into a `Base` *value* (`Base s = d;`) keeps
  only the `Base` sub-object, so later calls on `s` behave as a plain `Base`.

The hierarchy under test:

```cpp
struct Base {
    virtual int who() { return 1; }   // Base::who        -> tag 1
    int nonvirt()     { return 10; }  // Base::nonvirt     -> tag 10  (NON-virtual)
    virtual ~Base() {}
};
struct Derived : Base {
    int who() override { return 2; }  // Derived::who      -> tag 2
    int nonvirt()      { return 20; } // Derived::nonvirt  -> tag 20  (hides Base::nonvirt)
};
struct MoreDerived : Derived {
    int who() override { return 3; }  // MoreDerived::who  -> tag 3
};
```

Objects and handles in scope:

```cpp
Base b; Derived d; MoreDerived m;
Base*    pb_d = &d;   Base&  rb_d = d;   Base*  pb_b = &b;
Base*    pb_m = &m;   Derived* pd_m = &m;
Base     sliced = d;  Base* pb_d2 = &d;  Base&  rb_m = m;
```

## Task

Implement `void predict_tags(int out[12])` in `solve.cpp`. Store, in order, the tag
of the override that actually runs at each of these 12 call sites:

```
out[0]  = b.who();          //  object, static Base
out[1]  = d.who();          //  object, static Derived
out[2]  = pb_d->who();      //  virtual via Base*   pointing at a Derived
out[3]  = rb_d.who();       //  virtual via Base&   bound to a Derived
out[4]  = pb_b->who();      //  virtual via Base*   pointing at a Base
out[5]  = pb_m->who();      //  virtual via Base*   pointing at a MoreDerived
out[6]  = pd_m->who();      //  virtual via Derived* pointing at a MoreDerived
out[7]  = sliced.who();     //  sliced value object (copied from a Derived)
out[8]  = b.nonvirt();      //  non-virtual, static Base
out[9]  = d.nonvirt();      //  non-virtual, static Derived
out[10] = pb_d2->nonvirt(); //  non-virtual via Base* pointing at a Derived
out[11] = rb_m.who();       //  virtual via Base&   bound to a MoreDerived
```

You do not have to build the classes yourself — reason about each call and write the
predicted tag into the array. The driver in `main.cpp` prints your 12 tags.

## Example

If the calls were only `b.who()`, `pb_d->who()`, `pb_d2->nonvirt()`, the answer prefix
would be `1 2 10`:

- `b.who()` — value object of static type `Base` -> `Base::who` = 1.
- `pb_d->who()` — **virtual** through a `Base*` aimed at a `Derived` -> dynamic type
  wins -> `Derived::who` = 2.
- `pb_d2->nonvirt()` — **non-virtual** through a `Base*`; static type `Base` wins even
  though it points at a `Derived` -> `Base::nonvirt` = 10.

## What the gate checks

The driver (`main.cpp`) is compiled with `clang++ -O2 -std=c++20` together with your
`solve.cpp`, calls `predict_tags`, and prints the 12 tags. A hidden reference computes
the same vector by *actually executing* the 12 calls under real virtual-dispatch rules.
The gate is `exact_match`: your printed 12-integer vector must equal the reference's
vector exactly. Any single mis-predicted call fails the gate.
