#include <cstdio>
#include <set>
#include "sol.hpp"

struct Case { long n, stride, elem_bytes, line_bytes; };

// FIXED 20-case fixture: a spread of strides below, equal to, and above
// the line's element capacity L = line_bytes/elem_bytes, at two element
// sizes and two line sizes.
static const Case CASES[20] = {
    {100,  1, 4, 64}, {100,  2, 4, 64}, {100,  4, 4, 64}, {100,  8, 4, 64},
    {100, 16, 4, 64}, {100, 17, 4, 64}, {100, 32, 4, 64}, {100, 64, 4, 64},
    { 50,  3, 4, 64}, { 50,  5, 4, 64},
    {200,  1, 8, 64}, {200,  4, 8, 64}, {200,  8, 8, 64}, {200,  9, 8, 64},
    {200, 16, 8, 64},
    { 30,  7, 4, 32}, { 30,  8, 4, 32}, { 30,  9, 4, 32},
    {1000, 13, 4, 64}, {1000, 31, 4, 64},
};

// HARNESS ground truth (not learner code): actually walk the n accesses
// and count the distinct lines touched, via a std::set. This is the
// "simulator" the learner's closed form must agree with.
static long simulate_distinct_lines(long n, long stride, long elem_bytes, long line_bytes) {
    std::set<long> lines;
    for (long i = 0; i < n; i++) {
        long addr = i * stride * elem_bytes;
        lines.insert(addr / line_bytes);
    }
    return (long)lines.size();
}

int main() {
    int agree = 0;
    for (const auto& c : CASES) {
        long truth = simulate_distinct_lines(c.n, c.stride, c.elem_bytes, c.line_bytes);
        long got = distinct_lines_stride_walk(c.n, c.stride, c.elem_bytes, c.line_bytes);
        if (got == truth) agree++;
    }
    printf("agree=%d/20\n", agree);
    return 0;
}
