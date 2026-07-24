#include "sol.hpp"

// TODO: apply the One-Definition Rule to (kind, linkage, is_inline).
//   - Class/union types, enums, type aliases, and templates: return 1.
//   - Functions and variables with internal / no linkage: return 1.
//   - Functions and variables with external linkage: return 1 only if is_inline,
//     otherwise return 0 (they must have exactly one definition in the program).
int may_appear_in_multiple_tus(int kind, int linkage, int is_inline) {
    // your code here
    return 0;
}
