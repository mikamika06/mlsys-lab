#include "sol.hpp"

// TODO: use placement new (`::new (buf) Probe(a, b)`) to construct a Probe
// in buf, read its fields into *out_a/*out_b, then end its lifetime with an
// explicit destructor call. See sol.hpp.
void placement_lifecycle(void* buf, int a, double b, int* out_a, double* out_b) {
    (void)buf; (void)a; (void)b;
    *out_a = 0;
    *out_b = 0.0;
    // your code here
}
