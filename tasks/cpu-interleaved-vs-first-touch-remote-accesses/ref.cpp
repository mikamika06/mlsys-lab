#include "sol.hpp"
#include <unordered_map>

void count_remote_accesses(const Access* trace, int n, int num_nodes,
                            long* first_touch_remote, long* interleaved_remote) {
    std::unordered_map<long, int> owner;  // page -> first-touch home node
    long ft_remote = 0, il_remote = 0;

    for (int k = 0; k < n; k++) {
        int thread = trace[k].thread;
        int my_node = thread % num_nodes;
        long page = trace[k].addr / PAGE_BYTES;

        auto it = owner.find(page);
        int home;
        if (it == owner.end()) {
            home = my_node;
            owner[page] = home;
        } else {
            home = it->second;
        }
        if (home != my_node) ft_remote++;

        int il_home = (int)(page % num_nodes);
        if (il_home != my_node) il_remote++;
    }

    *first_touch_remote = ft_remote;
    *interleaved_remote = il_remote;
}
