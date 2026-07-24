#include "sol.hpp"

// TODO: for each candidate tile, reset_cache() then touch_byte() every
// address of its 3*B*B footprint, `passes` times in a row; record the
// total misses, then return the id of the tile with fewer misses. See
// sol.hpp.
int pick_resident_tile(int tile_b0, int tile_b1, int passes, long* out_misses) {
    (void)tile_b0; (void)tile_b1; (void)passes;
    out_misses[0] = 0;
    out_misses[1] = 0;
    // your code here
    return 0;
}
