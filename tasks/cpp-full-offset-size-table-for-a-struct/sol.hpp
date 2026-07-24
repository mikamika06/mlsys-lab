#pragma once

enum class FieldType { Bool, Char, Short, Int, Long, LongLong, Float, Double, Pointer };

struct FieldLayout {
    int offset;
    int size;
};

// Predict, for a struct whose members appear in this exact order with
// these types, each member's (offset, size) under this platform's real
// struct-layout rules: natural alignment (alignment == size for every one
// of these basic types), fields packed sequentially with padding inserted
// so each field starts at a multiple of its own alignment, and trailing
// padding so the struct's total size is a multiple of its largest
// member's alignment.
//
// Write one FieldLayout per input field into `out` (which has room for
// `n` entries, one per entry of `fields`), and write the struct's total
// sizeof (including tail padding) to *total_size_out.
void struct_layout(const FieldType* fields, int n, FieldLayout* out, int* total_size_out);
