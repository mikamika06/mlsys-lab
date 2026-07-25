#include "sol.hpp"

void count_local_remote(const long* addrs, int n, long page_bytes,
                         const int* node_of_page, int num_pages,
                         int home_node, long* out) {
    (void)num_pages;
    long local = 0, remote = 0;
    for (int i = 0; i < n; i++) {
        long page = addrs[i] / page_bytes;
        int node = node_of_page[page];
        if (node == home_node) local++;
        else remote++;
    }
    out[0] = local;
    out[1] = remote;
}
