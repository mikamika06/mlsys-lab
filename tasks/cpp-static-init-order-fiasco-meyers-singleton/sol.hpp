#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// get_b_value() must return 42, and — this is the whole point — it must be
// SAFE to call from ANOTHER translation unit's own namespace-scope
// (dynamic) initializer, no matter which translation unit's static
// initializers the linker/runtime happens to run first. The C++ standard
// gives NO guarantee about relative initialization order of namespace-scope
// globals defined in different translation units ("static initialization
// order fiasco").
//
// The only architecture immune to that is a MEYERS SINGLETON: wrap the
// value in a function-local `static` inside get_b_value() —
//     int get_b_value() { static int value = /* computed once */; return value; }
// — never a plain namespace-scope global that some other function merely
// reads. A function-local static is guaranteed to be constructed the FIRST
// TIME CONTROL PASSES THROUGH ITS DECLARATION, wherever that first call
// comes from — even from another translation unit's own dynamic
// initializer running before this TU's own globals have initialized.
// ============================================================================
int get_b_value();
