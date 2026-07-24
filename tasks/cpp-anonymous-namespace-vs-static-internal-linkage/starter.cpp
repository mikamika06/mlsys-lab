#include "sol.hpp"

// TODO: return 1 if `static` at namespace scope gives the same observable
// internal-linkage behavior as an anonymous namespace for category `c`,
// or 0 if the two diverge (illegal, or different semantics). Dispatch on
// `c` using the C++20 rules described in task.md.
int is_equivalent(Category c) {
    (void)c;
    return 0;  // placeholder: claims every category diverges
}
