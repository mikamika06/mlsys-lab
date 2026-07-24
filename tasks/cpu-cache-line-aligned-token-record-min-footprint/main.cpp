#include <cstdio>
#include <cstdint>
#include <cstring>
#include <list>
#include <unordered_map>
#include "sol.hpp"

// ============================================================================
// FIXED driver.
//
// Deterministic fully-associative LRU cache: 64 lines, 64 bytes/line
// (4096-byte cache total). touch_byte(addr) maps addr -> line (addr / 64)
// and counts a miss whenever that line is not already resident.
// ============================================================================
namespace {
constexpr int LINE_BYTES = 64;
constexpr int CACHE_LINES = 64;

std::list<long> g_lru;                                   // MRU at front
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
        g_map[line] = g_lru.begin();
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

// ============================================================================
// Workload: N tokens, repeatedly swept HOT_PASSES times, touching only the
// hot fields (id, count, flags) each pass. Whether the whole array's hot
// working set fits inside the 4096-byte cache depends entirely on how large
// sizeof(record) is -- which is exactly what the candidate's layout controls.
// ============================================================================
constexpr int N = 70;
constexpr int HOT_PASSES = 40;

int main() {
    size_t rsz = record_size();
    size_t o_id = offset_id();
    size_t o_count = offset_count();
    size_t o_flags = offset_flags();
    size_t o_name = offset_name();
    size_t o_ts = offset_ts();

    size_t bytes = static_cast<size_t>(N) * rsz + LINE_BYTES; // slack for alignment
    unsigned char* raw = new unsigned char[bytes];
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (addr + (LINE_BYTES - 1)) & ~static_cast<uintptr_t>(LINE_BYTES - 1);
    unsigned char* base = reinterpret_cast<unsigned char*>(aligned);

    // Initialize every field of every record (not measured -- no touch_byte here).
    for (int i = 0; i < N; ++i) {
        unsigned char* rp = base + static_cast<size_t>(i) * rsz;
        uint32_t idv = static_cast<uint32_t>(i);
        uint32_t cnt0 = 0;
        uint8_t fl0 = 0;
        uint64_t ts0 = 0;
        std::memcpy(rp + o_id, &idv, sizeof(idv));
        std::memcpy(rp + o_count, &cnt0, sizeof(cnt0));
        std::memcpy(rp + o_flags, &fl0, sizeof(fl0));
        std::memset(rp + o_name, 0, 24);
        std::memcpy(rp + o_ts, &ts0, sizeof(ts0));
    }

    reset_cache();
    uint64_t checksum = 0;
    for (int p = 0; p < HOT_PASSES; ++p) {
        for (int i = 0; i < N; ++i) {
            unsigned char* rp = base + static_cast<size_t>(i) * rsz;

            touch_byte(reinterpret_cast<long>(rp + o_id));
            uint32_t idv;
            std::memcpy(&idv, rp + o_id, sizeof(idv));

            touch_byte(reinterpret_cast<long>(rp + o_count));
            uint32_t cntv;
            std::memcpy(&cntv, rp + o_count, sizeof(cntv));
            cntv += 1;
            std::memcpy(rp + o_count, &cntv, sizeof(cntv));

            touch_byte(reinterpret_cast<long>(rp + o_flags));
            uint8_t flv;
            std::memcpy(&flv, rp + o_flags, sizeof(flv));
            flv = static_cast<uint8_t>(flv ^ 1);
            std::memcpy(rp + o_flags, &flv, sizeof(flv));

            checksum += idv + cntv + flv;
        }
    }

    printf("record_size=%zu\n", rsz);
    printf("misses=%ld\n", miss_count());
    printf("checksum=%llu\n", static_cast<unsigned long long>(checksum));

    delete[] raw;
    return 0;
}
