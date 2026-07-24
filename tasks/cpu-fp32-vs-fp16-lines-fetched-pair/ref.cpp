#include <set>
#include "sol.hpp"

static long lines_for_width(long base, int n, int width, int line_bytes) {
    std::set<long> lines;
    for (int i = 0; i < n; i++) {
        long addr = base + (long)i * width;
        lines.insert(addr / line_bytes);
    }
    return (long)lines.size();
}

void compare_fp32_fp16_lines(long base, int n, int line_bytes, long* out) {
    out[0] = lines_for_width(base, n, 4, line_bytes);
    out[1] = lines_for_width(base, n, 2, line_bytes);
}
