#pragma once

// Fixed record (real C++ — real sizeof/alignment, no modeled ABI).
struct Result {
    int id;
    float val;
};

// ============================================================================
// LEARNER implements this in solve.cpp — it MUST return the constructed
// Result BY VALUE (never a reference to something local to the function,
// direct or indirect).
// ============================================================================
Result get_result(int id, float val);
