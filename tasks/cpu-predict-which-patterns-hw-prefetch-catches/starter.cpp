#include "sol.hpp"

// TODO: a pattern is caught (1) iff EVERY consecutive stride in its
// trace is the same nonzero value AND that stride's magnitude is < 4096
// bytes (one page); otherwise it's not caught (0). See sol.hpp.
void classify_prefetch(const long* const* addrs, const int* lens, int num_patterns, int* out) {
    (void)addrs; (void)lens;
    // your code here
    for (int k = 0; k < num_patterns; k++) out[k] = 0;
}
