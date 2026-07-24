# Standard-layout & trivially-copyable flags (12 structs)

## Context

Two C++ ABI properties decide how a type may be laid out in memory and copied:

- **Standard-layout** (`std::is_standard_layout`) governs whether a type has a
  predictable, C-compatible layout (so `offsetof`, `memcpy` into a matching C
  struct, and reinterpreting the first member are well-defined). A class is
  *not* standard-layout if any of these hold: it has virtual functions or
  virtual bases; it has a reference data member; its non-static data members do
  not all share the **same access control**; a base class *and* the derived
  class both declare non-static data members; or the first non-static data
  member has the **same type as a base class**. User-defined constructors,
  destructors, and static data members do **not** affect standard-layout.

- **Trivially-copyable** (`std::is_trivially_copyable`) governs whether an
  object can be copied byte-for-byte with `memcpy`. A class is *not*
  trivially-copyable if it (or one of its members/bases) has a virtual
  function, a user-provided copy/move constructor, a user-provided copy/move
  assignment, or a user-provided (non-trivial) destructor. Access control and
  base-vs-derived data placement do **not** affect trivial-copyability.

The two properties are independent: a struct can be standard-layout but not
trivially-copyable (e.g. a user-defined destructor), or trivially-copyable but
not standard-layout (e.g. a reference member).

Below are the twelve structs under test. Reason about each one.

```cpp
struct S1  { int a; double b; };                                  // plain aggregate
struct S2  { public: int a; private: int b; };                    // mixed access control
struct S3  { int a; virtual void f(); };                          // has a virtual function
struct S4  { int a; S4() {} S4(const S4&) {} };                   // user-defined copy ctor
struct S5  { int& r; };                                           // reference data member
struct B6  { int a; }; struct S6 : B6 { int b; };                 // base AND derived hold data
struct B7  {}; struct S7 : B7 { int a; };                         // empty base, data in derived only
struct S8  { int a; ~S8() {} };                                   // user-defined destructor
struct S9  { int a; static int s; };                              // extra member is static
struct I10 { I10() {} I10(const I10&) {} };
struct S10 { I10 x; int a; };                                     // member has non-trivial copy
struct I11 { int a; int b; }; struct S11 { I11 arr[3]; int c; };  // array of standard-layout struct
struct B12 {}; struct S12 : B12 { B12 b; int a; };                // first member has same type as base
```

## Task

Implement `classify` (declared in `sol.hpp`):

```cpp
void classify(int out[24]);
```

Fill `out[0..23]` with two bits per struct, in order `S1..S12`, **standard-layout
bit first**:

- `out[2*(k-1)]     = 1` if `Sk` is standard-layout, else `0`
- `out[2*(k-1) + 1] = 1` if `Sk` is trivially-copyable, else `0`

So `out = [ SL(S1), TC(S1), SL(S2), TC(S2), ..., SL(S12), TC(S12) ]`.

You may reason the bits out by hand from the rules above, or paste the struct
definitions into `solve.cpp` and read the answers from `<type_traits>` — both
approaches produce the same 24-bit vector. The driver in `main.cpp` calls your
`classify` and prints the result; do not edit `main.cpp`.

## Example

For a smaller vector of two structs `[T1, T2]` where

```cpp
struct T1 { int a; double b; };          // standard-layout, trivially-copyable
struct T2 { int a; virtual void f(); };  // neither
```

the expected output bits would be:

```
1 1 0 0
```

(`SL(T1)=1, TC(T1)=1, SL(T2)=0, TC(T2)=0`).

## What the gate checks

The grader compiles `main.cpp` with your `solve.cpp` using
`clang++ -O2 -std=c++20`, runs it, and compares the printed output to the
reference (`ref.cpp`, which reads the properties from `std::type_traits`). The
gate is **exact match**: every one of the 24 bits (and the two summary counts
derived from them) must equal the reference. The all-zeros starter fails; a
correct 24-bit classification passes.
