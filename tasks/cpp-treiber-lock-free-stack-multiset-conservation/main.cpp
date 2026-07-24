#include <cstdio>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include "sol.hpp"

// FIXED driver. P producer threads each push a known block of DISTINCT integers;
// C consumer threads pop concurrently until the stack is drained AND every
// producer has finished. The union of pushed values is exactly {0 .. TOTAL-1}.
//
// A correct lock-free stack CONSERVES the multiset: whatever the thread
// schedule, the sorted popped values are exactly {0 .. TOTAL-1}. So the numbers
// printed below are DETERMINISTIC even though the interleaving is not. A racy or
// empty stack loses / duplicates items and prints different numbers.

int main() {
    const int P     = 4;          // producer threads
    const int C     = 4;          // consumer threads
    const int PER   = 25000;      // values pushed per producer
    const int TOTAL = P * PER;    // 100000 distinct values, 0 .. TOTAL-1

    TreiberStack stack;
    std::atomic<int> producers_done{0};

    std::vector<std::thread> threads;
    threads.reserve(P + C);

    // Producers: producer p pushes p*PER + i for i in [0, PER).
    for (int p = 0; p < P; ++p) {
        threads.emplace_back([&, p] {
            for (int i = 0; i < PER; ++i)
                stack.push(p * PER + i);
            producers_done.fetch_add(1, std::memory_order_seq_cst);
        });
    }

    // Consumers: pop until the stack is empty AND all producers have finished.
    // Each consumer records into its own vector (no shared-write race).
    std::vector<std::vector<int>> got(C);
    for (int c = 0; c < C; ++c) {
        threads.emplace_back([&, c] {
            int v;
            for (;;) {
                if (stack.pop(v)) {
                    got[c].push_back(v);
                } else if (producers_done.load(std::memory_order_seq_cst) == P) {
                    // Producers finished => no more pushes will ever happen.
                    // One final drain attempt; if it fails the stack is truly
                    // empty and this consumer is done.
                    if (stack.pop(v)) { got[c].push_back(v); continue; }
                    break;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    for (auto& t : threads) t.join();

    // Aggregate all popped values and test the conservation invariant.
    std::vector<int> all;
    all.reserve(TOTAL);
    for (auto& g : got)
        all.insert(all.end(), g.begin(), g.end());
    std::sort(all.begin(), all.end());

    const int n = (int)all.size();
    long long sum = 0;
    long long xr  = 0;
    int conserved = (n == TOTAL) ? 1 : 0;   // is the sorted multiset {0..TOTAL-1}?
    for (int i = 0; i < n; ++i) {
        sum += all[i];
        xr  ^= all[i];
        if (i >= TOTAL || all[i] != i) conserved = 0;
    }

    printf("count=%d\n", n);
    printf("sum=%lld\n", sum);
    printf("xor=%lld\n", xr);
    printf("conserved=%d\n", conserved);
    return 0;
}
