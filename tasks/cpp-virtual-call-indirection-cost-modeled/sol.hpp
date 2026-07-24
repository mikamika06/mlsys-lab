#pragma once
// ---------------------------------------------------------------------------
// Virtual-call indirection cost (modeled).
//
// A sequence of virtual calls is described by two parallel arrays of length n:
//   obj_id[i] : identity of the receiver object at call site i
//   slot[i]   : which virtual method (vtable slot index) is invoked at site i
//
// COST MODEL. One "indirect load" = one dependent, pointer-chasing memory read.
// A naive virtual call performs TWO of them:
//   (1) vptr load       : read the object's hidden vtable pointer (offset 0).
//   (2) vtable-slot load : read the target function pointer out of the vtable.
// A devirtualized (statically resolved / inlined) call performs NEITHER.
// ---------------------------------------------------------------------------

// 1) Naive virtual loop: 2 indirect loads per call, always.  Returns the total.
long naive_virtual_loads(const int* obj_id, const int* slot, int n);

// 2) Register-cached virtual loop. The compiler keeps the last-loaded vptr and
//    the last-loaded function pointer live in registers:
//      - the vptr load is skipped iff obj_id[i] == obj_id[i-1];
//      - the vtable-slot load is skipped iff obj_id[i] == obj_id[i-1]
//        AND slot[i] == slot[i-1] (same function pointer already in a register).
//    The first call (i == 0) pays both loads.  Returns the total indirect loads.
long cached_virtual_loads(const int* obj_id, const int* slot, int n);

// 3) Fully devirtualized loop: every call is resolved at compile time, so no
//    indirect loads occur at all.  Returns the total (which is 0).
long devirtualized_loads(const int* obj_id, const int* slot, int n);
