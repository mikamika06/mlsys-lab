#include "sol.hpp"

// TODO: implement the legality check described in task.md:
//   1. restrict-qualified pointers are always safe.
//   2. an access that is exactly in-place (same base byte offset, same
//      size) is always safe.
//   3. otherwise, safe iff no vector block's WRITE bytes overlap any LATER
//      block's READ bytes.
bool is_safe_to_vectorize(const KernelSpec& spec) {
    (void)spec;
    // your code here
    return true;
}
