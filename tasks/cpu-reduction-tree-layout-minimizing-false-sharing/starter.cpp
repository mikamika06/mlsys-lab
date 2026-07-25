#include "sol.hpp"

// BROKEN: no padding. stride stays 8, so all 16 threads' slots pack
// into two 64-byte lines (8 slots each) -- round 0's 8 concurrent
// writers alone thrash both lines repeatedly.
int slot_pad_bytes() {
    return 0;
}
