#pragma once

// Category of the construct in a snippet. The exact meaning of the numeric
// fields (a, b, width, flag) depends on the category and is documented in
// task.md.
enum Category {
    SIGNED_ADD   = 0,  // signed integer addition a + b in a `width`-bit type
    UNSIGNED_ADD = 1,  // unsigned integer addition in a `width`-bit type
    ARRAY_IDX    = 2,  // access element index b of an array of length a
    UNINIT_READ  = 3,  // read of an automatic variable (flag: 1 init, 0 uninit)
    NULL_DEREF   = 4,  // pointer dereference (flag: 1 pointer is null, 0 valid)
    SHIFT        = 5,  // shift a value of a `width`-bit type left by b positions
    TYPE_PUN     = 6,  // access via cast (flag: 0 reinterpret_cast, 1 memcpy)
};

// One classified code snippet, encoded as structured data.
struct Snippet {
    int       op;     // one of Category
    long long a;      // operand 1 / array length
    long long b;      // operand 2 / index / shift amount
    int       width;  // bit width of the integer type involved (8, 16, 32, 64)
    int       flag;   // category-specific boolean (see enum / task.md)
};

// Return 1 if the snippet exhibits undefined behavior under the C++20 rules,
// or 0 if its behavior is well-defined.
int classify_ub(const Snippet& s);
