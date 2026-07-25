#include <cstdio>
#include <list>
#include <unordered_map>
#include <vector>
#include "sol.hpp"

// FIXED driver + FIXED TLB model. 16384-byte pages, 20-entry fully-
// associative LRU TLB. Deterministic matrix contents (no rand()/time()):
// value(i, j) = (i * 131 + j * 977) % 1009.

namespace {

constexpr long long kPageBytes = 16384;
constexpr int kTlbEntries = 20;

struct Tlb {
    std::list<long long> lru;                                             // MRU-first
    std::unordered_map<long long, std::list<long long>::iterator> pos;
    int misses = 0;

    void reset() {
        lru.clear();
        pos.clear();
        misses = 0;
    }

    void access(long long page) {
        auto it = pos.find(page);
        if (it != pos.end()) {
            lru.erase(it->second);
            lru.push_front(page);
            pos[page] = lru.begin();
            return;
        }
        ++misses;
        if (static_cast<int>(lru.size()) >= kTlbEntries) {
            long long victim = lru.back();
            lru.pop_back();
            pos.erase(victim);
        }
        lru.push_front(page);
        pos[page] = lru.begin();
    }
};

Tlb g_tlb;

double value(int i, int j) {
    return static_cast<double>((i * 131 + j * 977) % 1009);
}

}  // namespace

void tlb_reset() { g_tlb.reset(); }

void touch_page(const void* p) {
    long long addr = reinterpret_cast<long long>(p);
    g_tlb.access(addr / kPageBytes);
}

int tlb_miss_count() { return g_tlb.misses; }

int main() {
    constexpr int R = 256;
    constexpr int C = 256;
    constexpr int ld = C;

    std::vector<double> data(static_cast<size_t>(R) * ld);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            data[static_cast<size_t>(i) * ld + j] = value(i, j);

    tlb_reset();
    double sum = sum_matrix_reordered(data.data(), R, C, ld);

    printf("R=%d C=%d page_bytes=%lld tlb_entries=%d\n", R, C, kPageBytes, kTlbEntries);
    printf("sum=%.1f\n", sum);
    printf("tlb_misses=%d\n", tlb_miss_count());
    return 0;
}
