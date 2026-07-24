#pragma once
#include <cstddef>

// Multiple-inheritance hierarchy (Itanium C++ ABI, LP64). B1 and B2 are each
// independently polymorphic -- each carries its OWN vptr, at the start of
// its own base subobject. B3 is not polymorphic (no vtable). Derived
// combines all three, plus one member of its own.
struct B1 {
    virtual ~B1() = default;
    long x1;
};
struct B2 {
    virtual ~B2() = default;
    int    x2;
    double y2;
};
struct B3 {
    char  x3;
    short y3;
};
struct Derived : B1, B2, B3 {
    int extra;
};

// Fill offs[] with the THIS-POINTER ADJUSTMENT (in bytes) applied when
// converting a Derived* to each base pointer type -- i.e. where that base's
// subobject actually starts inside a Derived object:
//   offs[0] = adjustment for Derived* -> B1*   (B1 is the PRIMARY base)
//   offs[1] = adjustment for Derived* -> B2*   (a SECONDARY base)
//   offs[2] = adjustment for Derived* -> B3*   (a SECONDARY base)
//   offs[3] = sizeof(Derived)
//
// B1, being the primary base, shares Derived's own vptr slot at offset 0,
// so its adjustment is 0. B2 has its OWN vptr -- it cannot also start at
// offset 0, where B1's vptr already lives -- so its subobject starts later,
// after B1's. B3 comes after that.
//
// A `static_cast<B2*>(derived_ptr)` is NOT a no-op pointer copy: the
// compiler silently adds exactly this adjustment every time such a cast
// appears in real code (this is the "this-pointer adjustment thunk"). A
// `reinterpret_cast` would NOT apply it, which is why it is the wrong tool
// for casting across bases in a multiple-inheritance hierarchy.
void base_offsets(std::size_t offs[4]);
