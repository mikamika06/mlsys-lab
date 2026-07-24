#include "sol.hpp"

PadResult min_padding_for_n_vars(long var_bytes, int line_bytes, long n) {
    long rem = var_bytes % (long)line_bytes;
    long padding = ((long)line_bytes - rem) % (long)line_bytes;
    long stride = var_bytes + padding;
    PadResult r;
    r.padding_bytes = padding;
    r.stride_bytes = stride;
    r.total_bytes = stride * n;
    return r;
}
