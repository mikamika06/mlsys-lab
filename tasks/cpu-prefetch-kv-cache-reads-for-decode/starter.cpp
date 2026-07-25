#include "sol.hpp"

// TODO: see sol.hpp -- simulate_decode_pass() replays the T-record scan
// with a given prefetch_distance and counts demand misses;
// choose_best_prefetch_distance() searches distances 0..max_distance and
// returns the one with fewest misses (smallest on a tie).
long simulate_decode_pass(int T, int rec_bytes, int prefetch_distance) {
    (void)T; (void)rec_bytes; (void)prefetch_distance;
    // your code here
    return 0;
}

int choose_best_prefetch_distance(int T, int rec_bytes, int max_distance) {
    (void)T; (void)rec_bytes; (void)max_distance;
    // your code here
    return 0;
}
