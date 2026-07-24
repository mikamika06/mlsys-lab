#include <cstdio>
#include "sol.hpp"

// PROVIDED. Probe's real constructors/destructor and the global counters.
Probe::Probe(int tag_, double payload_) : tag(tag_), payload(payload_) { ++g_direct_count; }
Probe::Probe(const Probe& other) : tag(other.tag), payload(other.payload) { ++g_copy_count; }
Probe::Probe(Probe&& other) noexcept : tag(other.tag), payload(other.payload) { ++g_move_count; }
Probe::~Probe() {}

long g_direct_count = 0;
long g_copy_count = 0;
long g_move_count = 0;

// FIXED driver. Do not edit. Calls make_probe for four fixed (tag, payload)
// cases, printing the resulting fields and the REAL construction counts
// observed via Probe's own constructors.
int main() {
    struct Case { int tag; double payload; };
    Case cases[] = {{1, 3.5}, {42, -1.25}, {7, 0.0}, {99, 128.5}};

    for (const auto& c : cases) {
        g_direct_count = 0;
        g_copy_count = 0;
        g_move_count = 0;
        Probe p = make_probe(c.tag, c.payload);
        long total = g_direct_count + g_copy_count + g_move_count;
        printf("tag=%d payload=%.6f direct=%ld copy=%ld move=%ld total=%ld\n",
               p.tag, p.payload, g_direct_count, g_copy_count, g_move_count, total);
    }
    return 0;
}
