#pragma once

// Fixed scalar type tags — the LP64 primitives this task lays out.
enum class FieldType { Bool, Char, Short, Int, Long, LongLong, Float, Double, Pointer };

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Given `n` field types in declaration order, compute the layout a real
// C++ compiler would give a class/struct with exactly those members, under
// natural alignment (align == size for every type here): insert inter-field
// padding so each field starts at a multiple of its own alignment, AND tail
// padding so the total size is a multiple of the class's strictest member's
// alignment. Write each field's offset into out_offsets[i] and return the
// total sizeof.
// ============================================================================
int compute_layout(const FieldType* fields, int n, int* out_offsets);
