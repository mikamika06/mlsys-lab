#include "sol.hpp"

// Only even tids ever write (odd tids are always the "read-only" side of
// a combine), and among even tids the smallest stride that keeps every
// pair's addresses in distinct 64-byte lines is 32 bytes (64 / 2): line
// = floor(tid*32/64) = floor(tid/2), which is already injective over
// the even tids 0,2,4,...,14 -> 0,1,2,...,7. 16 bytes is NOT enough
// (floor(tid/4) collides for tid=0 and tid=2). 32 = 8 (counter) + 24
// (padding).
int slot_pad_bytes() {
    return 24;
}
