#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU TLB: S=64 sets, W=4 ways (256
// entries), reconfigurable page size. Real hardware TLB behaviour is not
// reproducible across machines, so this model is what the driver grades
// against, not any real CPU's TLB.
struct Tlb {
    long page_size;
    int nsets = 64, ways = 4;
    std::vector<std::list<long>> sets;
    long misses = 0;

    explicit Tlb(long ps) : page_size(ps), sets(nsets) {}

    void access(long addr) {
        long page = addr / page_size;
        auto& s = sets[(int)(page % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == page) { s.erase(it); s.push_front(page); return; }
        }
        misses++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(page);
    }
};

static Tlb* TLB = nullptr;

void touch(long byte_addr) { TLB->access(byte_addr); }
void reset_tlb(long page_size) { delete TLB; TLB = new Tlb(page_size); }
long miss_count() { return TLB ? TLB->misses : 0; }

// FIXED driver, two fixed scenarios. Scenario 1's working set (300 pages
// * 4096 B = 1,228,800 B) exceeds the 4 KiB-page TLB's reach (256
// entries * 4096 B = 1,048,576 B), so replaying it thrashes under 4 KiB
// pages but fits entirely inside a single 2 MiB page. Scenario 2's
// working set (200 pages * 4096 B = 819,200 B) fits inside the 4 KiB
// TLB's reach too, so it should NOT thrash under 4 KiB pages either --
// it still collapses to a single 2 MiB page.
int main() {
    struct Scenario { long base, stride; int count, passes; };
    static const Scenario scenarios[] = {
        {0, 4096, 300, 3},
        {0, 4096, 200, 4},
    };

    for (const auto& sc : scenarios) {
        long out[2] = {0, 0};
        tlb_miss_pair(sc.base, sc.stride, sc.count, sc.passes, out);
        printf("count=%d stride=%ld passes=%d miss_4k=%ld miss_2m=%ld\n",
               sc.count, sc.stride, sc.passes, out[0], out[1]);
    }
    return 0;
}
