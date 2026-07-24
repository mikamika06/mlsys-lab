#include "sol.hpp"
#include <cstdio>
#include <set>

namespace {
std::set<long> g_lines;
}

void reset_touch() { g_lines.clear(); }
void touch(long byte_addr) { g_lines.insert(byte_addr / 64); }
long touched_line_count() { return static_cast<long>(g_lines.size()); }

struct Config { int n; int stride_elems; int width; };

int main() {
    static const Config configs[] = {
        {1000, 1, 4},    // fully packed contiguous walk
        {1000, 2, 4},    // skip every other element
        {1000, 16, 4},   // one element per cache line (16*4=64)
        {256, 1, 8},
        {256, 8, 8},     // one element per line (8*8=64)
        {64, 32, 4},     // very sparse
    };
    for (const auto& c : configs) {
        double eff = byte_efficiency(c.n, c.stride_elems, c.width);
        printf("%.9f\n", eff);
    }
    return 0;
}
