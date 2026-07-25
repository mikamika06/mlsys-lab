#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include "sol.hpp"

// ============================================================================
// FIXED driver: per-64-byte-line "last writer" coherence model.
// ============================================================================
namespace {
std::unordered_map<long, int> g_last_writer; // line -> thread id
long g_invalidations = 0;
} // namespace

void reset_coherence() {
    g_last_writer.clear();
    g_invalidations = 0;
}

void write_counter(int thread_id, long addr) {
    long line = addr / 64;
    auto it = g_last_writer.find(line);
    if (it != g_last_writer.end()) {
        if (it->second != thread_id) {
            ++g_invalidations;
        }
        it->second = thread_id;
    } else {
        g_last_writer[line] = thread_id;
    }
}

long invalidation_count() { return g_invalidations; }

// ============================================================================
// Deterministic round-robin workload: ROUNDS times, every thread (in
// order 0..NUM_THREADS-1) bumps all NUM_BINS of its own local counters by
// 1 -- simulating NUM_THREADS worker threads each tallying one token
// into every one of their local histogram bins per round.
// ============================================================================
constexpr int ROUNDS = 20;

int main() {
    size_t stride = thread_block_stride();
    size_t total_bytes = static_cast<size_t>(NUM_THREADS) * stride + 64; // slack for alignment

    unsigned char* raw = new unsigned char[total_bytes];
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (addr + 63) & ~static_cast<uintptr_t>(63);
    unsigned char* base = reinterpret_cast<unsigned char*>(aligned);
    std::memset(base, 0, static_cast<size_t>(NUM_THREADS) * stride);

    reset_coherence();
    for (int round = 0; round < ROUNDS; ++round) {
        for (int t = 0; t < NUM_THREADS; ++t) {
            unsigned char* block = base + static_cast<size_t>(t) * stride;
            for (int b = 0; b < NUM_BINS; ++b) {
                int64_t* p = reinterpret_cast<int64_t*>(block + static_cast<size_t>(b) * sizeof(int64_t));
                write_counter(t, reinterpret_cast<long>(p));
                *p += 1;
            }
        }
    }

    int64_t checksum = 0;
    for (int t = 0; t < NUM_THREADS; ++t) {
        unsigned char* block = base + static_cast<size_t>(t) * stride;
        for (int b = 0; b < NUM_BINS; ++b) {
            int64_t* p = reinterpret_cast<int64_t*>(block + static_cast<size_t>(b) * sizeof(int64_t));
            checksum += *p;
        }
    }

    printf("stride=%zu\n", stride);
    printf("total_bytes=%zu\n", static_cast<size_t>(NUM_THREADS) * stride);
    printf("invalidations=%ld\n", invalidation_count());
    printf("checksum=%lld\n", static_cast<long long>(checksum));

    delete[] raw;
    return 0;
}
