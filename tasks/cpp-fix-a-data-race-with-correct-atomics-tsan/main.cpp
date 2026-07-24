#include <cstdio>
#include <thread>
#include <vector>
#include "sol.hpp"

// FIXED driver. T threads each call increment() ITERS times on the SAME
// shared counter, with no synchronization between them beyond `increment`
// itself. A correct atomic read-modify-write can NEVER lose an update, so
// the final count is EXACTLY T * ITERS on every run, no matter how the OS
// schedules the threads. A load()+store() race loses updates under real
// concurrent execution -- at this thread count and iteration count, on real
// multi-core hardware, it essentially always comes out wrong.
int main() {
    const int T = 8;
    const long ITERS = 200000;

    SharedCounter counter;
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; t++) {
        threads.emplace_back([&counter]() {
            for (long i = 0; i < ITERS; i++) increment(counter);
        });
    }
    for (auto& th : threads) th.join();

    long final_value = counter.value.load();
    long expected = (long)T * ITERS;
    printf("final=%ld expected=%ld ok=%d\n",
           final_value, expected, (final_value == expected) ? 1 : 0);
    return 0;
}
