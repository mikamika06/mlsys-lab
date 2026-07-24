#pragma once

// ============================================================================
// Fixed record types (real C++ — real compiler layout, three different
// field orderings/paddings around one or more `double` fields).
// ============================================================================
struct RecordA { char c; double d1; double d2; };
struct RecordB { double d; char c; int i; double d2; };
struct RecordC { int i; double d; float f; double d2; };

// ============================================================================
// LEARNER implements these three in solve.cpp.
//
// For every element of the array, add 1.0 to EVERY field of type `double`,
// in place — leave every non-double field untouched. This is the C++ side
// of a zero-copy numpy buffer-protocol binding: the array is mutated
// through the pointer, not rebuilt.
// ============================================================================
void mutate_a(RecordA* arr, int n);
void mutate_b(RecordB* arr, int n);
void mutate_c(RecordC* arr, int n);
