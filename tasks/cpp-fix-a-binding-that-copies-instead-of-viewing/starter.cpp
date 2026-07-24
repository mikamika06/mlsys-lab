#include <cstring>
#include "sol.hpp"

// BUG: allocates a fresh owned buffer and memcpy's the payload into it,
// instead of returning a view into buf's own memory. Reads happen to look
// right at first, but writes through the returned pointer never reach
// `buf` -- this is exactly "copies instead of views".
double* view_payload(unsigned char* buf, int header_size, int n) {
    double* owned = new double[n];
    std::memcpy(owned, buf + header_size, (size_t)n * sizeof(double));
    return owned;
}
