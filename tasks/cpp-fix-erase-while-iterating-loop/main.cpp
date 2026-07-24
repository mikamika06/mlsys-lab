#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Do not edit. Two adjacent negative values (id 2 and id 3)
// are the case that exposes the skip bug: a correct implementation must
// remove BOTH of them, a buggy one leaves one behind.
int main() {
    std::vector<DataNode> nodes = {
        {1, 10, 100},
        {2, -5, 200},
        {3, -10, 300},
        {4, 20, 400},
        {5, -1, 500},
    };

    filter_nodes(nodes);

    printf("sizeof=%d count=%d\n", (int)sizeof(DataNode), (int)nodes.size());
    for (const auto& n : nodes) {
        printf("id=%d value=%d next=%ld\n", n.id, n.value, n.next);
    }
    return 0;
}
