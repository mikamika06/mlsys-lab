#include "sol.hpp"

// TODO: search pad in {0, 4, 8, ..., max_pad_bytes}; for each, measure
// the two-pass single-row gather's miss count against a fresh
// (line_bytes, sets, ways) cache via touch()/reset_cache()/miss_count();
// return the pad with the fewest misses (smallest pad breaks ties). See
// sol.hpp.
int choose_padding_bytes(int H, int row_bytes, int line_bytes, int sets,
                          int ways, int max_pad_bytes) {
    (void)H; (void)row_bytes; (void)line_bytes;
    (void)sets; (void)ways; (void)max_pad_bytes;
    // your code here
    return 0;
}
