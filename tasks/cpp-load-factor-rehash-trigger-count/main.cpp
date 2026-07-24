#include "sol.hpp"
#include <cstdio>
#include <cstddef>
#include <vector>

static void run_case(const std::vector<long>& inserts, double max_load, int init_b) {
    SimResult r = simulate_hash_map(inserts.data(), static_cast<int>(inserts.size()), max_load, init_b);
    int real_size = static_cast<int>(sizeof(HashNode));
    printf("%d %d %d\n", r.rehash_count, r.hash_node_size, r.hash_node_size == real_size ? 1 : 0);
}

int main() {
    {
        std::vector<long> v = {1, 2, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13};
        run_case(v, 1.0, 4);
    }
    {
        std::vector<long> v;
        for (long i = 0; i < 100; i++) v.push_back(i);
        run_case(v, 0.75, 8);
    }
    {
        std::vector<long> v = {5, 5, 5, 5, 5};
        run_case(v, 0.5, 2);
    }
    {
        std::vector<long> v;
        for (long i = 0; i < 20; i++) v.push_back(i);
        for (long i = 0; i < 20; i++) v.push_back(i);
        run_case(v, 2.0, 2);
    }
    return 0;
}
