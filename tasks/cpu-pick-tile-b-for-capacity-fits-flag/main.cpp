#include <cstdio>
#include <cstdint>
#include <list>
#include <new>
#include <unordered_map>
#include "sol.hpp"

// ============================================================================
// FIXED driver: deterministic fully-associative LRU cache, CAPACITY_BYTES
// bytes of it, LINE_BYTES-byte lines.
// ============================================================================
namespace {
constexpr int LINE_BYTES = 64;
constexpr long CAPACITY_BYTES = 4096;
constexpr int CACHE_LINES = CAPACITY_BYTES / LINE_BYTES; // 64

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

constexpr int ELEM_SIZE = 4; // sizeof(float)

int main() {
    int B = derive_tile_b(CAPACITY_BYTES, ELEM_SIZE);
    long count = B > 0 ? static_cast<long>(B) * B : 0;

    // Three independently 64-byte-aligned tiles.
    static float* tile_a = nullptr;
    static float* tile_b = nullptr;
    static float* tile_c = nullptr;
    tile_a = new (std::align_val_t(64)) float[count > 0 ? count : 1];
    tile_b = new (std::align_val_t(64)) float[count > 0 ? count : 1];
    tile_c = new (std::align_val_t(64)) float[count > 0 ? count : 1];
    for (long i = 0; i < count; ++i) {
        tile_a[i] = static_cast<float>(i);
        tile_b[i] = static_cast<float>(i) * 2.0f;
        tile_c[i] = 0.0f;
    }

    reset_cache();
    // Pass 1: touch every element of all three tiles once (cold).
    for (long i = 0; i < count; ++i) touch_byte(reinterpret_cast<long>(&tile_a[i]));
    for (long i = 0; i < count; ++i) touch_byte(reinterpret_cast<long>(&tile_b[i]));
    for (long i = 0; i < count; ++i) touch_byte(reinterpret_cast<long>(&tile_c[i]));
    long misses_after_pass1 = miss_count();

    // Pass 2: touch the SAME elements again, same order. If the three
    // tiles genuinely fit in cache, everything is still resident and this
    // pass contributes zero additional misses.
    for (long i = 0; i < count; ++i) touch_byte(reinterpret_cast<long>(&tile_a[i]));
    for (long i = 0; i < count; ++i) touch_byte(reinterpret_cast<long>(&tile_b[i]));
    for (long i = 0; i < count; ++i) touch_byte(reinterpret_cast<long>(&tile_c[i]));
    long misses_after_pass2 = miss_count();

    int fits = (misses_after_pass2 == misses_after_pass1) ? 1 : 0;

    printf("B=%d\n", B);
    printf("misses_pass1=%ld\n", misses_after_pass1);
    printf("misses_pass2=%ld\n", misses_after_pass2);
    printf("fits=%d\n", fits);

    delete[] tile_a;
    delete[] tile_b;
    delete[] tile_c;
    return 0;
}
