#include "sol.hpp"

// TODO: touch every line in [base, base+nbytes) via
// touch(line_addr(base, line_bytes, k)) for k in [0, nbytes/line_bytes),
// in order, exactly once each -- see sol.hpp.
void temporal_memset(long base, long nbytes, int line_bytes) {
    (void)base; (void)nbytes; (void)line_bytes;
    // your code here
}
