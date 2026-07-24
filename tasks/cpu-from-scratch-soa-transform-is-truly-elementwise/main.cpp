#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache. Real hardware cache behaviour
// is not reproducible across machines, so this model -- 64-byte lines,
// 8 sets, 4-way (2048 bytes total) -- is what the driver grades against,
// not the CPU's real cache.
struct Level {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;
    long touches = 0;

    Level(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

    void access(long addr) {
        touches++;
        long line = addr / line_bytes;
        auto& s = sets[(int)(line % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); s.push_front(line); return; }
        }
        misses++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(line);
    }
};

static Level CACHE(64, 8, 4);

void touch(long byte_addr) { CACHE.access(byte_addr); }
long touch_count() { return CACHE.touches; }
long miss_count() { return CACHE.misses; }
void reset_stats() { CACHE = Level(64, 8, 4); }

// FIXED driver. Runs aos_field_to_soa over 4 fixed (record_count,
// field_count, field_index) scenarios -- 4-field records extracting the
// first field, the same records extracting the last field, wider 8-field
// records extracting a middle field, and narrow 2-field records -- each
// against its own fresh cache/counters, and prints the touch count, the
// miss count, and whether the transform was genuinely elementwise
// (touch_count == 2 * record_count).
int main() {
    struct Scenario { int record_count, field_count, field_index; };
    static const Scenario scenarios[] = {
        {2000, 4, 0},
        {2000, 4, 3},
        {500, 8, 2},
        {5000, 2, 1},
    };

    for (const auto& s : scenarios) {
        reset_stats();
        long aos_base = 0;
        long soa_out_base = (long)s.record_count * s.field_count * 4;  // right after the AoS region
        aos_field_to_soa(aos_base, soa_out_base, s.record_count, s.field_count, s.field_index);
        long expected = 2L * s.record_count;
        printf("rc=%d fc=%d fi=%d touches=%ld misses=%ld elementwise=%d\n",
               s.record_count, s.field_count, s.field_index,
               touch_count(), miss_count(),
               (touch_count() == expected) ? 1 : 0);
    }
    return 0;
}
