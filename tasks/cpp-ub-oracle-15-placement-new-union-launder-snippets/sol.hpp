#pragma once

// One step of a sequential trace of operations on a single raw storage
// location.
enum OpKind { ALLOCATE, PLACEMENT_NEW, DTOR, ACCESS };

struct Op {
    OpKind kind;
    int type;        // an arbitrary type tag (0=int, 1=float, 2=string-like
                      // non-trivial type); meaningful for PLACEMENT_NEW/ACCESS
    int is_const;     // PLACEMENT_NEW: is the constructed object const?
    int is_trivial;   // PLACEMENT_NEW: is the constructed object trivially
                       // destructible (0 = needs an explicit dtor call
                       // before this storage may be reused, like std::string)
    int is_write;      // ACCESS: is this a write?
    int laundered;      // ACCESS: was the pointer passed through std::launder
                          // first?
};

// Replay ops[0..n) against ONE raw storage location and decide whether the
// trace contains undefined behavior, under these rules:
//
//   ALLOCATE       -- (re)start with empty, unconstructed storage.
//   PLACEMENT_NEW  -- UB if an object is currently alive AND that object is
//                     NOT trivially destructible (its storage was reused
//                     without calling its destructor first). Otherwise,
//                     construct the new object; if the object being
//                     overwritten (alive or not, as long as one has
//                     EVER occupied this storage) was `const`, the ORIGINAL
//                     pointer becomes stale for every object constructed at
//                     this address from now on, until an access launders it.
//   DTOR           -- UB if no object is currently alive (double-destroy /
//                     destroying nothing).
//   ACCESS         -- UB if no object is alive (out of lifetime); UB if the
//                     access `type` doesn't match the CURRENTLY alive
//                     object's type; UB if `is_write` is true but the
//                     current object is const; UB if the pointer is stale
//                     (see above) and this access is not `laundered`. A
//                     `laundered` access clears staleness for every FUTURE
//                     access to the current object, not just this one.
//
// Return 1 if the trace is UB, 0 if every operation in it is well-defined.
int classify_ub(const Op* ops, int n);
