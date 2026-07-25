#include <cstdio>
#include <vector>
#include "sol.hpp"

static std::vector<long> range_step(long start, long stop, long step) {
    std::vector<long> v;
    for (long x = start; x < stop; x += step) v.push_back(x);
    return v;
}

int main() {
    struct Scenario { std::vector<long> addrs; int line_bytes, slots; };
    std::vector<Scenario> scenarios;
    scenarios.push_back({range_step(0, 64, 1), 64, 2});
    scenarios.push_back({range_step(0, 128, 2), 64, 2});
    scenarios.push_back({{0, 64, 128, 192, 256, 320}, 64, 3});
    scenarios.push_back({{0, 4, 8, 12, 16, 20, 64, 68, 72}, 32, 1});
    scenarios.push_back({range_step(0, 512, 8), 64, 4});

    for (const auto& s : scenarios) {
        long out[2] = {0, 0};
        wc_flush_stats(s.addrs.data(), (int)s.addrs.size(), s.line_bytes, s.slots, out);
        printf("n=%d line_bytes=%d slots=%d full=%ld partial=%ld\n",
               (int)s.addrs.size(), s.line_bytes, s.slots, out[0], out[1]);
    }
    return 0;
}
