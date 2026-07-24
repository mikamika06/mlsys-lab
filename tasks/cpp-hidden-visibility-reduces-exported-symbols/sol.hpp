#pragma once

// Predict how many of `n` function declarations end up as EXPORTED dynamic
// symbols in a real shared library, given:
//
//   global_hidden : whether the library was compiled with -fvisibility=hidden.
//   is_static[i]  : whether declaration i has `static` (internal) linkage.
//   attr[i]       : declaration i's explicit visibility attribute --
//                     0 = __attribute__((visibility("default")))
//                     1 = __attribute__((visibility("hidden")))
//                     2 = no explicit attribute (inherits the global default)
//
// Rules (real Itanium C++ ABI / GCC-and-Clang visibility semantics):
//   - `static` (internal linkage) is NEVER exported, no matter what
//     visibility attribute is also present.
//   - An explicit visibility attribute on an externally-linked declaration
//     always wins over the global -fvisibility flag.
//   - With no explicit attribute, an externally-linked declaration inherits
//     the global default: exported when global_hidden is false, hidden
//     when global_hidden is true.
//
// Return the number of declarations (out of n) that end up exported.
int count_exported_symbols(bool global_hidden, const int* is_static, const int* attr, int n);
