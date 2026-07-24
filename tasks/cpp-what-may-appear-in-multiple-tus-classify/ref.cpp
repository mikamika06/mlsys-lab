#include "sol.hpp"

int may_appear_in_multiple_tus(int kind, int linkage, int is_inline) {
    // Types, enumerations, type aliases, and templates: an identical definition
    // may (and often must) appear in every TU that uses them.
    if (kind == KIND_CLASS || kind == KIND_ENUM ||
        kind == KIND_ALIAS || kind == KIND_TEMPLATE)
        return 1;

    // Remaining kinds are ordinary functions and variables.
    // Internal linkage (or no linkage): a distinct entity in each TU, so it may
    // be defined once per TU.
    if (linkage != LINK_EXTERNAL)
        return 1;

    // External linkage: only inline functions / inline variables may be defined
    // in more than one TU.
    if (is_inline)
        return 1;

    // Non-inline function or variable with external linkage: exactly one
    // definition is allowed in the whole program.
    return 0;
}
