#include "sol.hpp"

// TODO: for each access i, find the closest earlier index j touching
// the same 64-byte line, count the distinct lines touched strictly
// between j and i, and track the maximum such count over the whole
// trace. See sol.hpp.
long max_reuse_distance(const long* addrs, int n) {
    (void)addrs; (void)n;
    // your code here
    return 0;
}
