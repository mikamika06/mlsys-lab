#include <cstdio>
#include <unordered_map>
#include "sol.hpp"

// PROVIDED. Deterministic cache-line-ownership model (see sol.hpp).
namespace {
std::unordered_map<long, int> g_line_owner;
long g_invalidations = 0;
}  // namespace

void report_write(int thread_id, long byte_addr) {
    long line = byte_addr / CACHE_LINE_BYTES;
    auto it = g_line_owner.find(line);
    if (it == g_line_owner.end()) {
        g_line_owner[line] = thread_id;
        return;
    }
    if (it->second != thread_id) {
        ++g_invalidations;
        it->second = thread_id;
    }
}

long total_invalidations() { return g_invalidations; }

void reset_invalidations() {
    g_line_owner.clear();
    g_invalidations = 0;
}

// FIXED driver. Do not edit.
int main() {
    const int NUM_THREADS = 8;
    const int NUM_INCREMENTS = 100;

    long aos = simulate_aos_invalidations(NUM_THREADS, NUM_INCREMENTS);
    long padded = simulate_padded_invalidations(NUM_THREADS, NUM_INCREMENTS);

    printf("aos=%ld\n", aos);
    printf("padded=%ld\n", padded);
    return 0;
}
