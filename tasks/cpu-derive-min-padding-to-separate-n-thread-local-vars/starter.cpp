#include "sol.hpp"

// TODO: padding = (line_bytes - (var_bytes mod line_bytes)) mod line_bytes;
// stride = var_bytes + padding; total = stride * n. See sol.hpp -- mind
// the OUTER mod, a var that already fills whole lines needs 0 padding.
PadResult min_padding_for_n_vars(long var_bytes, int line_bytes, long n) {
    (void)var_bytes; (void)line_bytes; (void)n;
    // your code here
    PadResult r;
    r.padding_bytes = 0;
    r.stride_bytes = 0;
    r.total_bytes = 0;
    return r;
}
