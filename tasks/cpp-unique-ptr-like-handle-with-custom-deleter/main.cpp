#include <cstdio>
#include <utility>
#include "sol.hpp"

// FIXED driver. Each scenario resets the global release counter, exercises the
// RAII handle through one ownership path, then records how many releases the
// custom deleter performed. A correct move-only handle releases every acquired
// resource exactly once.

// Used by the exception scenario: the handle must be released while the stack
// unwinds out of this function.
static void may_throw() {
    GpuHandle h(gpu_acquire());
    throw 42;                    // h is destroyed during unwinding
}

int main() {
    // Scenario 1 — normal scope exit: one acquire, one release.
    g_release_count = 0;
    {
        GpuHandle h(gpu_acquire());
    }
    int s1 = g_release_count;                    // expect 1

    // Scenario 2 — move construction: the moved-from handle must not release.
    g_release_count = 0;
    {
        GpuHandle a(gpu_acquire());
        GpuHandle b(std::move(a));               // b owns, a is empty
    }
    int s2 = g_release_count;                    // expect 1

    // Scenario 3 — move assignment over a LIVE handle: the old id is freed
    // immediately, the stolen id is freed at scope end.
    g_release_count = 0;
    {
        GpuHandle a(gpu_acquire());
        GpuHandle b(gpu_acquire());
        a = std::move(b);                        // frees a's old id now (1)
    }                                            // frees stolen id at end  (2)
    int s3 = g_release_count;                    // expect 2

    // Scenario 4 — exception unwinding: the handle is still released once.
    g_release_count = 0;
    try {
        may_throw();
    } catch (int) {
    }
    int s4 = g_release_count;                    // expect 1

    // Scenario 5 — reset(): releases now; the destructor must not double-free.
    g_release_count = 0;
    {
        GpuHandle a(gpu_acquire());
        a.reset();                               // frees now (1)
    }                                            // destructor: empty, no release
    int s5 = g_release_count;                    // expect 1

    int total = s1 + s2 + s3 + s4 + s5;          // expect 6

    printf("%d %d %d %d %d\n", s1, s2, s3, s4, s5);
    printf("total=%d\n", total);
    return 0;
}
