#include "sol.hpp"

// TODO: return 1 if this struct edit breaks the object ABI (the in-memory
// layout that already-compiled code relies on), else 0.
// Hint: compare size, align, the vptr flag, and every common field offset.
int abi_breaks(const Layout& old_v, const Layout& new_v) {
    // your code here
    return 0;
}
