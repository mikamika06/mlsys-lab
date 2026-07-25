#include <cstdio>
#include <list>
#include <unordered_map>
#include "sol.hpp"

// ============================================================================
// FIXED driver: deterministic fully-associative LRU cache, 64 lines of 64
// bytes each (4096 bytes total).
// ============================================================================
namespace {
constexpr int LINE_BYTES = 64;
constexpr int CACHE_LINES = 64;

std::list<long> g_lru;
std::unordered_map<long, std::list<long>::iterator> g_map;
long g_misses = 0;
} // namespace

void reset_cache() {
    g_lru.clear();
    g_map.clear();
    g_misses = 0;
}

void touch_byte(long addr) {
    long line = addr / LINE_BYTES;
    auto it = g_map.find(line);
    if (it != g_map.end()) {
        g_lru.erase(it->second);
        g_lru.push_front(line);
        it->second = g_lru.begin();
        return;
    }
    ++g_misses;
    if (static_cast<int>(g_lru.size()) >= CACHE_LINES) {
        long evict = g_lru.back();
        g_lru.pop_back();
        g_map.erase(evict);
    }
    g_lru.push_front(line);
    g_map[line] = g_lru.begin();
}

long miss_count() { return g_misses; }

// 2048 floats * 4 bytes == 8192 bytes == 128 lines, twice the cache's
// 64-line capacity -- re-reading the whole array a second time cannot
// reuse anything left over from the first read.
constexpr int N = 2048;

alignas(64) static float g_x[N];

int main() {
    for (int i = 0; i < N; ++i) {
        g_x[i] = static_cast<float>(i % 13) * 0.3f + 1.0f;
    }

    reset_cache();
    float sum = 0.0f, sumsq = 0.0f;
    compute_stats(g_x, N, &sum, &sumsq);
    long misses = miss_count();

    printf("sum=%.6f\n", sum);
    printf("sumsq=%.6f\n", sumsq);
    printf("misses=%ld\n", misses);
    return 0;
}
