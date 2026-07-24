#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. 4 independent pointer-chase chains of lengths 5, 5, 5, 3
// (nodes 0-4, 5-9, 10-14, 15-17). Every chain is internally fully
// sequential (each node depends on the previous one in its own chain), but
// the 4 chains have no dependencies between them.
int main() {
    std::vector<int> from_v, to_v;
    auto add_chain = [&](int start, int len) {
        for (int i = 0; i < len - 1; ++i) {
            from_v.push_back(start + i);
            to_v.push_back(start + i + 1);
        }
    };
    add_chain(0, 5);
    add_chain(5, 5);
    add_chain(10, 5);
    add_chain(15, 3);

    int n = 18;
    int num_edges = static_cast<int>(from_v.size());
    int result = mlp_degree(n, from_v.data(), to_v.data(), num_edges);

    printf("mlp_degree=%d\n", result);
    return 0;
}
