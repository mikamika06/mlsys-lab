#include "sol.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::vector<std::vector<int>> cases;
    cases.push_back({});
    cases.push_back({42});
    cases.push_back({1, 2, 3});
    cases.push_back({1, 2, 3, 4});
    cases.push_back({1, 2, 3, 4, 5});
    cases.push_back({1, 2, 3, 4, 5, 6, 7, 8, 9});
    {
        std::vector<int> zeros_ones;
        for (int i = 0; i < 10; i++) zeros_ones.push_back(0);
        for (int i = 0; i < 10; i++) zeros_ones.push_back(1);
        cases.push_back(zeros_ones);
    }
    {
        std::vector<int> range1000;
        for (int i = 0; i < 1000; i++) range1000.push_back(i);
        cases.push_back(range1000);
    }

    for (const auto& c : cases) {
        const int* data = c.empty() ? nullptr : c.data();
        long long r = simd_sum(data, static_cast<int>(c.size()));
        printf("%lld\n", r);
    }
    return 0;
}
