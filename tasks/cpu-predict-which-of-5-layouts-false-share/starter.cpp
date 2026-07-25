#include "sol.hpp"

// TODO: for each of the 5 fixed layouts (see sol.hpp), compute the 4
// thread byte addresses, then report true if any two land on the same
// line_bytes-sized cache line (floor(a_i / line_bytes) collide).
std::array<bool, 5> classify_layouts(long line_bytes) {
    (void)line_bytes;
    // your code here
    return {};
}
