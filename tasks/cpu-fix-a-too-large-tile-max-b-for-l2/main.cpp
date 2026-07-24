#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache modelling L2: 64-byte lines,
// 128 sets, 4-way -> 128*4*64 = 32768 bytes capacity. Real hardware
// cache behaviour isn't reproducible across machines, so this model is
// the sole source of every "does it fit" answer the driver prints.
struct Level {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;

    Level(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

    void access(long addr) {
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

static Level CACHE(64, 128, 4);

static void sweep(long nbytes) {
    long nlines = (nbytes + 63) / 64;
    for (long k = 0; k < nlines; k++) CACHE.access(k * 64);
}

// Runs two back-to-back sweeps of the B tiles' combined byte footprint
// over a FRESH cache: the first (cold-fill) pass causes any
// capacity/associativity self-eviction if the footprint is too big for
// L2; the second (reuse) pass is where that shows up as misses. The
// tile set is resident iff the second pass adds zero new misses.
static bool fits_l2(long B) {
    long bytes = 3L * B * B * 4;
    if (bytes < 0 || bytes / 64 > 5000000L) return false;  // obviously too big, skip simulating
    CACHE = Level(64, 128, 4);
    sweep(bytes);
    long before = CACHE.misses;
    sweep(bytes);
    long after = CACHE.misses;
    return after == before;
}

// FIXED driver. L2_BYTES matches the modelled cache's own capacity
// exactly (128 sets * 4 ways * 64B = 32768 bytes).
int main() {
    const long L2_BYTES = 32768;

    long b = max_tile_b_for_l2(L2_BYTES);
    int fits_b = fits_l2(b) ? 1 : 0;
    int fits_b_plus_1 = fits_l2(b + 1) ? 1 : 0;

    printf("b=%ld fits_b=%d fits_b_plus_1=%d\n", b, fits_b, fits_b_plus_1);
    return 0;
}
