// Fixed driver + deterministic cache-line model: buckets every touched
// byte address into its 64-byte line, so "bytes touched" really means
// "distinct 64-byte lines touched * 64" — the real mechanism by which
// AoS wastes bandwidth on a one-field workload (over-fetching whole
// lines full of fields you didn't ask for).
#include "sol.hpp"
#include <cstdio>
#include <set>

namespace {
std::set<long> g_lines;
}

void touch(long byte_addr) {
    g_lines.insert(byte_addr / 64);
}

static long measure(Layout layout, int n, int field_idx) {
    g_lines.clear();
    emit_access(layout, n, field_idx);
    return static_cast<long>(g_lines.size()) * 64;
}

int main() {
    const int N = 64;
    static const char* layout_names[3] = {"AoS", "SoA", "AoSoA"};
    static const int workloads[2] = {-1, 0}; // all fields, field 0 (id) only
    static const char* workload_names[2] = {"all", "field0"};

    for (int w = 0; w < 2; w++) {
        long costs[3];
        for (int L = 0; L < 3; L++) {
            costs[L] = measure(static_cast<Layout>(L), N, workloads[w]);
        }
        int winner = 0;
        for (int L = 1; L < 3; L++) {
            if (costs[L] < costs[winner]) winner = L;
        }
        printf("%s: AoS=%ld SoA=%ld AoSoA=%ld winner=%s\n", workload_names[w],
               costs[0], costs[1], costs[2], layout_names[winner]);
    }
    return 0;
}
