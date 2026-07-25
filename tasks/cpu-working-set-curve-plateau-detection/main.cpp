#include "sol.hpp"
#include <cstdio>

// FIXED driver, two scenarios. Each trace has a WARMUP prefix of
// distinct, never-repeated line addresses, followed by a STEADY region
// that cycles through K distinct 64-byte-line addresses (with the
// byte-level offset WITHIN each line also varying step to step, so
// distinct raw addresses outnumber distinct lines) repeated many times
// over. max_w is chosen so every window this task asks about stays fully
// inside the steady region -- a correct implementation never even looks
// at the warmup prefix.
int main() {
    {
        const int warmup = 15, K = 10, reps = 20;
        const int n = warmup + K * reps, max_w = 150;
        static long addrs[215];
        for (int i = 0; i < warmup; i++) addrs[i] = (long)(K + i) * 64;
        for (int i = warmup; i < n; i++) {
            int j = i - warmup;
            addrs[i] = (long)(j % K) * 64 + (long)(j % 3) * 4;
        }
        static int curve[150];
        int p = plateau_index(addrs, n, max_w, 64, curve);
        printf("scenario=1 K=%d n=%d plateau=%d curve1=%d curve_final=%d curve_at_plateau=%d\n",
               K, n, p, curve[0], curve[max_w - 1], curve[p - 1]);
    }
    {
        const int warmup = 8, K = 6, reps = 15;
        const int n = warmup + K * reps, max_w = 60;
        static long addrs[98];
        for (int i = 0; i < warmup; i++) addrs[i] = (long)(K + i) * 64;
        for (int i = warmup; i < n; i++) {
            int j = i - warmup;
            addrs[i] = (long)(j % K) * 64 + (long)(j % 3) * 4;
        }
        static int curve[60];
        int p = plateau_index(addrs, n, max_w, 64, curve);
        printf("scenario=2 K=%d n=%d plateau=%d curve1=%d curve_final=%d curve_at_plateau=%d\n",
               K, n, p, curve[0], curve[max_w - 1], curve[p - 1]);
    }
    return 0;
}
