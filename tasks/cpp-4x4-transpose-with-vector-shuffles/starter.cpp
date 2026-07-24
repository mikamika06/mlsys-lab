#include "sol.hpp"

// TODO: implement using vzipq_f32 shuffles from <arm_neon.h> (see sol.hpp for
// the exact composition). Right now `out` is left untouched (the caller's
// sentinel values), so it fails.
void transpose4x4(const float* in, float* out) {
    (void)in; (void)out;  // your code here
}
