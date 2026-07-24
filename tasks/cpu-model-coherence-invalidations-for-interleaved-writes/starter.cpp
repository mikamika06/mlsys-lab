#include "sol.hpp"

// TODO: track, per 64-byte line, the set of cores currently holding a
// valid copy; for each write, count the OTHER cores currently in that
// set (they get invalidated), then reset the set to just the writing
// core. See sol.hpp.
long count_invalidations(const WriteEvent* trace, int n) {
    (void)trace; (void)n;
    // your code here
    return 0;
}
