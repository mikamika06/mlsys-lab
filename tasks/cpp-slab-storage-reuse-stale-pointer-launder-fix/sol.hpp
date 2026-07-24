#pragma once

// Fixed slot type (real C++, real placement-new target).
struct Slot {
    int value;
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// A single storage slot gets reused for two consecutive objects:
//   1. Placement-new a Slot{5} into `storage`.
//   2. Placement-new a SECOND Slot{11} into the SAME storage — per
//      [basic.life], this ENDS the first Slot's lifetime and starts a new
//      one occupying the same bytes.
//   3. Return the value held there NOW (11).
//
// Read the final value ONLY through a pointer that is properly valid for
// the NEW object: the pointer returned by step 2's placement-new itself,
// or `std::launder(...)` applied to a pointer you re-derive from the
// storage address. Never read it through a pointer or a plain value you
// captured BEFORE step 2 — that pointer refers to an object that no
// longer exists, and a value cached from it is simply stale application
// data, not the slot's current contents.
// ============================================================================
int slab_reuse_demo();
