#include "sol.hpp"

// Fixed: return a pointer into buf's own memory -- a real zero-copy view.
double* view_payload(unsigned char* buf, int header_size, int n) {
    (void)n;
    return reinterpret_cast<double*>(buf + header_size);
}
