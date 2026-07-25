#include "sol.hpp"

// TODO: for each addrs[i], compute page = addrs[i] / page_bytes, look up
// node_of_page[page], and tally out[0] (local, node == home_node) or
// out[1] (remote, node != home_node). See sol.hpp.
void count_local_remote(const long* addrs, int n, long page_bytes,
                         const int* node_of_page, int num_pages,
                         int home_node, long* out) {
    (void)addrs; (void)n; (void)page_bytes; (void)node_of_page; (void)num_pages; (void)home_node;
    // your code here
    out[0] = 0;
    out[1] = 0;
}
