#pragma once

#include <cstddef>
#include <cstdint>

// Deliberately bad field order -- maximizes padding under the compiler's
// ordinary alignment rules (every field's offset is a multiple of its own
// size; the struct's total size is a multiple of its largest field's
// alignment).
struct NaiveStruct {
    bool flag_a;
    double value_x;
    char tag;
    int32_t count;
    bool flag_b;
    int64_t id;
    int16_t small_val;
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Define a struct -- anywhere inside solve.cpp, main.cpp never sees its
// definition -- containing the EXACT SAME 7 fields as NaiveStruct (one
// bool, one double, one char, one int32_t, a second bool, one int64_t,
// one int16_t -- field names don't matter, only their types), reordered
// to MINIMIZE sizeof(...) under the compiler's ordinary alignment rules.
//
// No #pragma pack, __attribute__((packed)), or alignas tricks -- the
// point is choosing a good ORDER, not disabling alignment. Every field
// must still sit at an offset that's a multiple of its own natural
// alignment, and the struct's total size must still be a multiple of its
// largest field's alignment.
//
// Return sizeof(your struct).
// ============================================================================
size_t packed_struct_size();
