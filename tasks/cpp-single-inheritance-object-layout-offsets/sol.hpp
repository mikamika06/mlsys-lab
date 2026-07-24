#pragma once
#include <cstddef>

// Single-inheritance, polymorphic class hierarchy (Itanium C++ ABI, LP64).
//
//   struct Base {
//       virtual ~Base();   // Base is polymorphic  ->  it carries a vptr
//       long  a;           // 8 bytes, align 8
//       int   b;           // 4 bytes, align 4
//   };
//   struct Derived : Base {
//       char   c;          // 1 byte,  align 1
//       short  d;          // 2 bytes, align 2
//       int    e;          // 4 bytes, align 4
//       double f;          // 8 bytes, align 8
//   };
//
// The two classes are defined for real below so you can reason about (and, if
// you wish, measure) their layout.
struct Base {
    virtual ~Base() = default;
    long a;
    int  b;
};

struct Derived : Base {
    char   c;
    short  d;
    int    e;
    double f;
};

// Fill offs[] with the byte offset of each of the following *inside a Derived
// object*, in exactly this order:
//   offs[0] = offset of the vptr
//   offs[1] = offset of Base::a
//   offs[2] = offset of Base::b
//   offs[3] = offset of Derived::c
//   offs[4] = offset of Derived::d
//   offs[5] = offset of Derived::e
//   offs[6] = offset of Derived::f
//   offs[7] = sizeof(Derived)
void derived_layout(std::size_t offs[8]);
