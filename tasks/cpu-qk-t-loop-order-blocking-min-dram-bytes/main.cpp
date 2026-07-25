#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache level -- THIS model, not the
// CPU's real cache, is the sole source of every miss count below.
struct Level {
    int nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;

    Level(int ns, int w) : nsets(ns), ways(w), sets(ns) {}

    void access(long line) {
        auto& s = sets[(int)(line % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); s.push_front(line); return; }
        }
        misses++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(line);
    }
};

static const int LINE_BYTES = 64;
static Level cache_(64, 8);  // 64 sets, 8-way, 64B lines -> 32768 bytes

void touch(long byte_addr) {
    cache_.access(byte_addr / LINE_BYTES);
}

// FIXED driver. Q and K are each S*d*elem_bytes = 128*64*4 = 32768 bytes
// -- Q and K TOGETHER (65536 bytes) are twice the size of the whole
// cache, so a full unblocked streaming pass over K genuinely cannot stay
// resident.
int main() {
    const int S = 128, d = 64, B = 16, elem_bytes = 4;
    qkt_access(S, d, B, elem_bytes);
    printf("misses=%ld\n", cache_.misses);
    return 0;
}
