#pragma once

struct Counters {
    long copies = 0;
    long moves = 0;
    long destructions = 0;
};
// Incremented by your Elem type's real copy/move/destructor calls; reset
// and defined by the harness (main.cpp) so the counts are trustworthy.
extern Counters g_counters;

struct GrowthCounts {
    long copies;
    long moves;
    long destructions;
    long total_alloc_bytes;
    long final_capacity;
};

// Push n_pushes elements, one at a time via emplace_back, onto a REAL
// std::vector<Elem>, where Elem is a type you define that is exactly
// element_size bytes (8 or 16, for the fixed cases this is graded
// against) and whose move constructor is conditionally noexcept:
//
//   Elem(Elem&& other) noexcept(move_is_noexcept) { ... }
//
// std::vector relocates existing elements during growth via MOVE
// construction only when the element's move constructor is noexcept;
// otherwise it falls back to COPY construction to preserve the strong
// exception guarantee (this is real std::vector<Elem> behavior via
// std::move_if_noexcept — nothing here is simulated).
//
// Reset g_counters before pushing. Return the REAL counts observed:
// copies/moves/destructions straight from g_counters, total_alloc_bytes =
// the sum of capacity() * sizeof(Elem) at every point capacity() changed,
// and final_capacity = the vector's capacity() after all n_pushes.
GrowthCounts simulate_vector_growth(int element_size, int n_pushes, bool move_is_noexcept);
