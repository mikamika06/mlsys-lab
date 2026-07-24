#pragma once

// Fixed vector-header record (real C++ sizeof, no modeled ABI).
struct VectorHeader {
    int size;
    int capacity;
    void* data;
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// grow_capacity(capacity, growth_factor) returns the NEW capacity a vector
// reallocates to when push_back finds size == capacity:
//
//     new_capacity = max(1, (int)(capacity * growth_factor))
//
// The cast to int truncates toward zero, exactly like C++'s own
// double -> int conversion (and Python's int()) for non-negative values.
// ============================================================================
int grow_capacity(int capacity, double growth_factor);
