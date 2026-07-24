#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache level. Real hardware cache
// behaviour is not reproducible across machines, so THIS model -- not the
// CPU's actual cache -- is the sole source of every miss count below.
struct Level {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;  // sets[i] = tags in this set, MRU-front
    long hits = 0, misses = 0;

    Level(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

    int idx_of(long line) const { return (int)(line % nsets); }

    bool has(long line) const {
        for (long v : sets[idx_of(line)]) if (v == line) return true;
        return false;
    }

    void invalidate(long line) {
        auto& s = sets[idx_of(line)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); return; }
        }
    }

    // Returns true on hit. On miss, inserts the line (evicting the LRU
    // line of its set if full) and reports the eviction via evicted/did_evict.
    bool access(long line, long& evicted, bool& did_evict) {
        auto& s = sets[idx_of(line)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) {
                s.erase(it);
                s.push_front(line);
                hits++;
                return true;
            }
        }
        misses++;
        did_evict = false;
        if ((int)s.size() >= ways) {
            evicted = s.back();
            s.pop_back();
            did_evict = true;
        }
        s.push_front(line);
        return false;
    }
};

static const int LINE_BYTES = 64;
static Level L1(LINE_BYTES, 8, 1);    // direct-mapped, 512 bytes
static Level L2(LINE_BYTES, 16, 4);   // 4-way, 4096 bytes
static Level L3(LINE_BYTES, 32, 8);   // 8-way, 16384 bytes

void touch(long byte_addr) {
    long line = byte_addr / LINE_BYTES;

    long ev1;
    bool did1;
    if (L1.access(line, ev1, did1)) return;  // L1 hit -> fully resolved

    long ev2;
    bool did2;
    bool l2_hit = L2.access(line, ev2, did2);
    if (!l2_hit) {
        long ev3;
        bool did3;
        bool l3_hit = L3.access(line, ev3, did3);
        if (!l3_hit && did3) {
            // L3 evicted a line: inclusion requires it can't survive in L2/L1.
            L2.invalidate(ev3);
            L1.invalidate(ev3);
        }
    }
    if (!l2_hit && did2) {
        // L2 evicted a line: inclusion requires it can't survive in L1.
        L1.invalidate(ev2);
    }
}

// FIXED driver. A 32x32 row-major float matrix (4096 bytes -- fits
// comfortably inside L2 and L3, but not L1) touched once, in whatever
// order access_pattern() chooses.
int main() {
    const int N = 32;
    access_pattern(N);
    printf("L1_misses=%ld L2_misses=%ld L3_misses=%ld\n", L1.misses, L2.misses, L3.misses);
    return 0;
}
