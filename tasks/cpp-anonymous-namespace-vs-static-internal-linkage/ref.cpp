#include "sol.hpp"

// Reference: canonical equivalence table under the C++20 rules described in
// task.md.
int is_equivalent(Category c) {
    switch (c) {
        case FREE_FUNCTION:      return 1;  // static void f(){} == namespace{ void f(){} }
        case FREE_VARIABLE:      return 1;  // static int x;     == namespace{ int x; }
        case CONST_VARIABLE:     return 1;  // const at namespace scope is already internal;
                                             // static is redundant but behaves the same
        case CLASS_TYPE:         return 0;  // `static class C {};` is ill-formed;
                                             // anonymous namespace is the only legal way
        case FUNCTION_TEMPLATE:  return 0;  // `static` cannot be applied to a function
                                             // template at namespace scope
        case CLASS_TEMPLATE:     return 0;  // `static` cannot be applied to a class template
        case INLINE_VARIABLE:    return 0;  // a static inline variable stays a single TU-local
                                             // definition; an inline var in an anonymous
                                             // namespace additionally becomes a distinct
                                             // internal-linkage entity per TU (ODR differs)
        case ENUM_TYPE:          return 0;  // `static` cannot be applied to an enum
                                             // type definition
        case TYPEDEF_ALIAS:      return 0;  // `static` cannot be applied to a
                                             // typedef/using alias
        case EXTERN_VARIABLE:    return 0;  // `static` and `extern` are contradictory on the
                                             // same declaration; `extern` inside an anonymous
                                             // namespace still ends up with internal linkage,
                                             // which is not the same declaration form
        default:                 return 0;
    }
}
