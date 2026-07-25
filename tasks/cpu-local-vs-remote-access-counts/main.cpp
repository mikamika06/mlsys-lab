#include <cstdio>
#include "sol.hpp"

// FIXED driver: 8 pages of 4096 bytes, alternately-ish placed across 2
// NUMA nodes, a thread pinned to node 0, and a deterministic 40-access
// trace (no rand()/time()) that visits every page a handful of times with
// a small in-page offset that never crosses a page boundary.
int main() {
    const long page_bytes = 4096;
    const int num_pages = 8;
    const int node_of_page[num_pages] = {0, 0, 1, 1, 0, 1, 0, 1};
    const int home_node = 0;

    const int n = 40;
    long addrs[n];
    for (int i = 0; i < n; i++) {
        int page = (i * 3 + 1) % num_pages;
        long offset = (i * 137) % page_bytes;
        addrs[i] = page * page_bytes + offset;
    }

    long out[2] = {-1, -1};  // sentinel: an empty starter leaves this untouched
    count_local_remote(addrs, n, page_bytes, node_of_page, num_pages, home_node, out);

    printf("local=%ld remote=%ld\n", out[0], out[1]);
    return 0;
}
