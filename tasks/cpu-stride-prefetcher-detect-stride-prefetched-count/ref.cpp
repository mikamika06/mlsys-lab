#include "sol.hpp"

long stride_prefetch_count(const int* stream_id, const long* addr, int n, int num_streams) {
    long last_addr[64] = {0};
    long last_delta[64] = {0};
    bool have_addr[64] = {false};
    bool have_delta[64] = {false};

    long prefetches = 0;
    for (int i = 0; i < n; i++) {
        int s = stream_id[i];
        long a = addr[i];
        if (!have_addr[s]) {
            last_addr[s] = a;
            have_addr[s] = true;
            continue;
        }
        long delta = a - last_addr[s];
        if (have_delta[s] && delta == last_delta[s]) {
            prefetches++;
        }
        last_delta[s] = delta;
        have_delta[s] = true;
        last_addr[s] = a;
    }
    (void)num_streams;
    return prefetches;
}
