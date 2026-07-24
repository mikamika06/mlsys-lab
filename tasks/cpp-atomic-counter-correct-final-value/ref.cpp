#include "sol.hpp"
#include <atomic>
#include <thread>
#include <vector>

long atomic_counter_final_value(int num_threads, int increments_per_thread) {
    std::atomic<long> counter{0};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&counter, increments_per_thread]() {
            for (int k = 0; k < increments_per_thread; k++) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) w.join();
    return counter.load();
}
