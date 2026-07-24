#pragma once

// Fixed element type (real C++ sizeof, no modeled ABI).
struct Item {
    int x;
    double y;
};

// What the caller learns about one push_back-loop run.
struct GrowthResult {
    int realloc_count;      // number of times v.data() changed address after the FIRST push
    long final_capacity;    // v.capacity() after the loop
    bool pointers_valid;    // did the pointer captured after the FIRST push survive to the end?
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Build a REAL std::vector<Item>. If `reserve_first` is true, call
// v.reserve(n_elements) BEFORE pushing anything. Then push_back n_elements
// real Items, one at a time (any deterministic values). Track how many
// times v.data() actually changes address across those pushes (the first
// push's own allocation doesn't count as a "re"-allocation — only address
// changes AFTER that count), and whether the address observed right after
// the first push is still v.data()'s address once the loop ends.
// ============================================================================
GrowthResult grow_vector(int n_elements, bool reserve_first);
