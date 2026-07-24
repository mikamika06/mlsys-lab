#include <cstdio>
#include "sol.hpp"

// Small POD types whose REAL sizeof() (this compiler's real ABI, padding and
// all) supplies the `bytes` field of each CONSTRUCT statement below -- the
// sizes are not hand-computed, they come from the compiler itself.
struct RecChar    { char a; };
struct RecInt     { int a; };
struct RecDouble  { double a; };
struct RecPad     { int a; char b; double c; };   // exercises alignment padding
struct RecPtrChar { void* p; char c; };

static void run_case(const Stmt* stmts, int n) {
    int ids[16];
    int count = 0;
    long total = run_trace(stmts, n, ids, &count);
    for (int i = 0; i < count; i++) printf("%d ", ids[i]);
    printf("| count=%d total=%ld\n", count, total);
}

// FIXED driver. Three scenarios: a throw nested three scopes deep (after one
// object was already destructed normally by an END), a trace that ends
// without ever throwing, and a throw with nothing yet live.
int main() {
    const Stmt a[] = {
        {BEGIN, 0, 0},
        {CONSTRUCT, 1, (int)sizeof(RecInt)},
        {BEGIN, 0, 0},
        {CONSTRUCT, 2, (int)sizeof(RecChar)},
        {END, 0, 0},                                 // normal dtor of 2
        {CONSTRUCT, 3, (int)sizeof(RecDouble)},
        {BEGIN, 0, 0},
        {CONSTRUCT, 4, (int)sizeof(RecPad)},
        {CONSTRUCT, 5, (int)sizeof(RecPtrChar)},
        {THROW, 0, 0},                                // unwinds 5, 4, 3, 1
        {CONSTRUCT, 6, (int)sizeof(RecInt)},          // unreachable
        {END, 0, 0},
    };
    run_case(a, (int)(sizeof(a) / sizeof(a[0])));

    const Stmt b[] = {
        {BEGIN, 0, 0},
        {CONSTRUCT, 10, (int)sizeof(RecInt)},
        {BEGIN, 0, 0},
        {CONSTRUCT, 11, (int)sizeof(RecDouble)},
        {END, 0, 0},
        {END, 0, 0},                                  // both closed normally, no throw
    };
    run_case(b, (int)(sizeof(b) / sizeof(b[0])));

    const Stmt c[] = {
        {BEGIN, 0, 0},
        {CONSTRUCT, 20, (int)sizeof(RecChar)},
        {END, 0, 0},                                  // 20 destructed normally
        {BEGIN, 0, 0},
        {THROW, 0, 0},                                 // nothing live to unwind
    };
    run_case(c, (int)(sizeof(c) / sizeof(c[0])));

    return 0;
}
