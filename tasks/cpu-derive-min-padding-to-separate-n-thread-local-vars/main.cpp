#include "sol.hpp"
#include <cstdio>

struct Scenario { long var_bytes; int line_bytes; long n; };

// FIXED driver. 6 scenarios covering: a small counter that needs most of
// a line padded on, a var that already exactly fills one line (needs NO
// padding), a var spanning two lines with a remainder, a 1-byte flag, a
// var that exactly fills two lines, and a smaller (32B) line size.
int main() {
    static const Scenario scenarios[] = {
        {8, 64, 4},
        {64, 64, 8},
        {100, 64, 3},
        {1, 64, 16},
        {128, 64, 2},
        {65, 32, 5},
    };

    for (const auto& s : scenarios) {
        PadResult r = min_padding_for_n_vars(s.var_bytes, s.line_bytes, s.n);
        printf("var=%ld line=%d n=%ld padding=%ld stride=%ld total=%ld\n",
               s.var_bytes, s.line_bytes, s.n,
               r.padding_bytes, r.stride_bytes, r.total_bytes);
    }
    return 0;
}
