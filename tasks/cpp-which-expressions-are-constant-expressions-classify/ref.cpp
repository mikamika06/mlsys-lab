#include "sol.hpp"
#include <cstddef>

// Reference oracle. The 12 bits are NOT hardcoded: each one is decided by the
// compiler itself. is_ce(lambda) is true iff the expression the lambda returns
// is a core constant expression, so this reference is verified by clang++'s own
// constant-expression evaluation.

namespace {

// ---- Entities referenced by the 12 expressions (see task.md) ----
constexpr int A = 7;                        // constexpr int
const int     B = 3;                        // const int, constant initializer
int           D = 5;                        // mutable global (runtime value)
const int     C = D + 1;                    // const int, NON-constant initializer
constexpr int arr[4] = {2, 4, 6, 8};        // constexpr array
constexpr int sq(int x) { return x * x; }   // constexpr function
int           rt(int x) { return x + D; }   // ordinary (non-constexpr) function

// Compile-time detector. probe_tag<V> names a valid type only when V is a
// constant expression, so the type-requirement inside the requires-expression
// is satisfied iff L{}() is a core constant expression. Captureless lambdas are
// default-constructible in C++20, so L{}() re-evaluates the wrapped expression
// in a constant context.
template <auto V> struct probe_tag {};
template <class L> constexpr bool is_ce(L) {
    return requires { typename probe_tag<L{}()>; };
}

} // namespace

unsigned classify_constexpr() {
    unsigned m = 0;
    auto set = [&](int i, bool ok) { if (ok) m |= (1u << i); };
    set(0,  is_ce([] { return A * 2; }));          // 1  constexpr int          -> yes
    set(1,  is_ce([] { return B + 1; }));          // 2  const int (const init) -> yes
    set(2,  is_ce([] { return C - 1; }));          // 3  const int (dyn init)   -> no
    set(3,  is_ce([] { return D + 0; }));          // 4  mutable global         -> no
    set(4,  is_ce([] { return sq(A); }));          // 5  constexpr fn, const arg-> yes
    set(5,  is_ce([] { return sq(D); }));          // 6  constexpr fn, dyn arg  -> no
    set(6,  is_ce([] { return rt(3); }));          // 7  non-constexpr fn       -> no
    set(7,  is_ce([] { return arr[3]; }));         // 8  constexpr[const index] -> yes
    set(8,  is_ce([] { return arr[D % 4]; }));     // 9  constexpr[dyn index]   -> no
    set(9,  is_ce([] { return sizeof(arr); }));    // 10 sizeof (unevaluated)   -> yes
    set(10, is_ce([] { return A > B ? A : B; }));  // 11 conditional, const ops -> yes
    set(11, is_ce([] { return A + D; }));          // 12 constexpr + runtime    -> no
    return m;
}
