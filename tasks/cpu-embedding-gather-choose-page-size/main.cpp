#include <cstdio>
#include <list>
#include <unordered_map>
#include "sol.hpp"

// FIXED driver + FIXED TLB model. Fully-associative LRU TLB, 16 entries.
// Real hardware TLB behaviour is not reproducible across machines, so this
// model -- not any real CPU's TLB -- is the sole source of every miss count.

namespace {

constexpr int kTlbEntries = 16;

struct Tlb {
    long page_bytes = 4096;
    std::list<long> lru;  // MRU-first
    std::unordered_map<long, std::list<long>::iterator> pos;
    int misses = 0;

    void reset(long pb) {
        page_bytes = pb;
        lru.clear();
        pos.clear();
        misses = 0;
    }

    void access(long addr) {
        long page = addr / page_bytes;
        auto it = pos.find(page);
        if (it != pos.end()) {
            lru.erase(it->second);
            lru.push_front(page);
            pos[page] = lru.begin();
            return;
        }
        ++misses;
        if (static_cast<int>(lru.size()) >= kTlbEntries) {
            long victim = lru.back();
            lru.pop_back();
            pos.erase(victim);
        }
        lru.push_front(page);
        pos[page] = lru.begin();
    }
};

Tlb g_tlb;

}  // namespace

void tlb_reset(long page_bytes) { g_tlb.reset(page_bytes); }

void touch_addr(long byte_addr) { g_tlb.access(byte_addr); }

int tlb_miss_count() { return g_tlb.misses; }

// FIXED driver: a deterministic scattered embedding gather trace (no
// rand()/time()) -- 64 row indices out of a 500-row table, row_bytes=256,
// so the whole table spans under 128000 bytes. Candidate page sizes are a
// small one (4096, far smaller than the table -- many distinct pages) and
// two huge ones (1 MiB and 2 MiB, both bigger than the whole table -- an
// intentional TIE the learner's tie-break rule must resolve).
int main() {
    const int n = 64;
    int indices[n];
    for (int i = 0; i < n; i++) indices[i] = (i * 97) % 500;  // scattered, deterministic

    const int row_bytes = 256;
    const long page_sizes[] = {4096, 1L << 20, 1L << 21};  // 4096, 1 MiB, 2 MiB
    const int p = 3;

    long best = choose_page_size(indices, n, row_bytes, page_sizes, p);
    printf("best_page_size=%ld\n", best);
    return 0;
}
