#pragma once
#include <vector>

// One mutation applied to a std::vector<long>.
enum OpKind {
    RESERVE,    // v.reserve(arg)
    PUSH_BACK,  // v.push_back(<fresh sentinel>)        (arg ignored)
    POP_BACK,   // v.pop_back()                         (arg ignored)
    INSERT,     // v.insert(v.begin() + arg, <fresh sentinel>)   (arg = position 0..size)
    CLEAR       // v.clear()                            (arg ignored)
};

struct Op {
    OpKind kind;
    int arg;
};

// A std::vector<long> is created with size n0 and capacity EXACTLY cap0
// (0 <= refIdx < n0 <= cap0), holding distinct sentinel values.
// A reference/pointer is taken to the element at index refIdx.
// The operations in `ops` are then applied in order, using the standard
// std::vector rules for iterator/reference invalidation.
//
// Return true iff, after all operations have run, the reference to the
// ORIGINAL element (the one that lived at index refIdx when the reference
// was taken) is still valid; return false if it was invalidated.
bool ref_survives(int n0, int cap0, int refIdx, const std::vector<Op>& ops);
