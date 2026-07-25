#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache: 64-byte lines, 32 sets, 4-way
// -> 8192 bytes total (the combined in+out working set below is 32768
// bytes, 4x the cache). Real hardware cache behaviour is not
// reproducible across machines, so this model is what the driver grades
// against, not the CPU's real cache.
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

static Level CACHE(64, 32, 4);

void touch(long byte_addr) { CACHE.access(byte_addr); }

// HARNESS baseline (not learner code): a correct 8x8-BLOCKED transpose,
// on the same real memory, for comparison against the learner's
// recursive cache-oblivious version. Both must reach the same miss count
// ballpark and, since they compute the same transpose, an identical
// checksum.
static void blocked_transpose(const float* in, float* out, int N, int B) {
    for (int ii = 0; ii < N; ii += B) {
        for (int jj = 0; jj < N; jj += B) {
            for (int row = ii; row < ii + B; row++) {
                for (int col = jj; col < jj + B; col++) {
                    out[col * N + row] = in[row * N + col];
                    touch(in_addr(N, row, col));
                    touch(out_addr(N, col, row));
                }
            }
        }
    }
}

static float IN[64 * 64];
static float OUT_BLOCKED[64 * 64];
static float OUT_CO[64 * 64];

static long checksum(const float* out, int N) {
    long s = 0;
    for (int i = 0; i < N * N; i++) s += (long)out[i] * (long)(i + 1);
    return s;
}

// FIXED driver. Fills a 64x64 matrix with exact, float-representable
// integer values, transposes it with the harness's blocked baseline and
// with the learner's co_transpose (each against its own fresh cache),
// and prints both miss counts plus a position-weighted checksum of the
// learner's output (sensitive to wrong VALUES, not just a wrong miss
// count -- summing the raw values alone wouldn't catch a wrong
// permutation, since transposition never changes what the multiset of
// values sums to).
int main() {
    const int N = 64;
    for (int i = 0; i < N * N; i++) IN[i] = (float)i;

    CACHE = Level(64, 32, 4);
    blocked_transpose(IN, OUT_BLOCKED, N, 8);
    long blocked_misses = CACHE.misses;

    CACHE = Level(64, 32, 4);
    co_transpose(IN, OUT_CO, N);
    long co_misses = CACHE.misses;

    printf("blocked_misses=%ld co_misses=%ld checksum=%ld\n",
           blocked_misses, co_misses, checksum(OUT_CO, N));
    return 0;
}
