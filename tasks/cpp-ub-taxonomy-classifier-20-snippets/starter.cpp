#include "sol.hpp"

// TODO: return 1 if the snippet exhibits undefined behavior under the C++20
// rules described in task.md, otherwise 0. Dispatch on s.op (a Category) and
// read the fields a, b, width, flag according to that category.
int classify_ub(const Snippet& s) {
    (void)s;
    return 0;  // placeholder: marks every snippet as "well-defined"
}
