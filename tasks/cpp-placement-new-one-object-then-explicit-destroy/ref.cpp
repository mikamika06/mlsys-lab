#include "sol.hpp"
#include <new>

void placement_lifecycle(void* buf, int a, double b, int* out_a, double* out_b) {
    Probe* p = ::new (buf) Probe(a, b);
    *out_a = p->a;
    *out_b = p->b;
    p->~Probe();
}
