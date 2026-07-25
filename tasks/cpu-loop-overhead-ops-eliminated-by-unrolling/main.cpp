#include <cstdio>

#include "sol.hpp"

// FIXED driver: several (N, U) pairs, including U == 1 (no unrolling,
// saved must be 0), U == N (fully unrolled), and pairs where U does not
// evenly divide N (exercises integer-division truncation).
struct Case { long long N, U; };

static const Case CASES[] = {
    {1024, 4},
    {1000, 3},
    {7, 7},
    {50, 1},
    {17, 5},
};

int main() {
    for (const auto& c : CASES) {
        printf("%lld\n", unroll_overhead_saved(c.N, c.U));
    }
    return 0;
}
