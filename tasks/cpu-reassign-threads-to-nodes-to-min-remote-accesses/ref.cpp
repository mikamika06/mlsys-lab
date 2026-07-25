#include "sol.hpp"
#include <vector>

namespace {
void backtrack(int t, int T, int N, const long* access_count, std::vector<int>& remaining,
               const std::vector<long>& totals, long remote_so_far, long& best) {
    if (remote_so_far >= best) return; // prune: can't possibly beat the best found so far
    if (t == T) {
        best = remote_so_far;
        return;
    }
    for (int n = 0; n < N; n++) {
        if (remaining[n] > 0) {
            remaining[n]--;
            long remote = totals[t] - access_count[t * N + n];
            backtrack(t + 1, T, N, access_count, remaining, totals, remote_so_far + remote, best);
            remaining[n]++;
        }
    }
}
} // namespace

long min_remote_accesses(int T, int N, int capacity, const long* access_count) {
    std::vector<int> remaining(N, capacity);
    std::vector<long> totals(T, 0);
    for (int t = 0; t < T; t++) {
        long s = 0;
        for (int n = 0; n < N; n++) s += access_count[t * N + n];
        totals[t] = s;
    }
    long best = 1LL << 60;
    backtrack(0, T, N, access_count, remaining, totals, 0, best);
    return best;
}
