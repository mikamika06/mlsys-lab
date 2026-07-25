#include "sol.hpp"
#include <vector>

long min_remote_accesses(int T, int N, int capacity, const long* access_count) {
    // your code here
    std::vector<int> remaining(N, capacity);
    long total_remote = 0;
    for (int t = 0; t < T; t++) {
        long total = 0;
        for (int n = 0; n < N; n++) total += access_count[t * N + n];
        int best_n = -1;
        long best_val = -1;
        for (int n = 0; n < N; n++) {
            if (remaining[n] > 0 && access_count[t * N + n] > best_val) {
                best_val = access_count[t * N + n];
                best_n = n;
            }
        }
        remaining[best_n]--;
        total_remote += total - access_count[t * N + best_n];
    }
    return total_remote;
}
