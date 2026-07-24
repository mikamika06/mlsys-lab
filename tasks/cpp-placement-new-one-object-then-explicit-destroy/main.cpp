#include <cstdio>
#include <cstddef>
#include "sol.hpp"

int g_ctor_count = 0;
int g_dtor_count = 0;

Probe::Probe(int a_, double b_) : a(a_), b(b_) { g_ctor_count++; }
Probe::~Probe() { g_dtor_count++; }

// FIXED driver. A correctly-aligned, poisoned raw buffer; calls
// placement_lifecycle once and prints the observed ctor/dtor counts, the
// recovered field values, and sizeof/alignof(Probe) (the real values, from
// the real compiler).
int main() {
    alignas(Probe) unsigned char buf[sizeof(Probe)];
    for (size_t i = 0; i < sizeof(buf); i++) buf[i] = 0xCC;  // poison

    g_ctor_count = 0;
    g_dtor_count = 0;
    int out_a = -1;
    double out_b = -1.0;
    placement_lifecycle(buf, 7, 2.5, &out_a, &out_b);

    printf("ctor_count=%d dtor_count=%d\n", g_ctor_count, g_dtor_count);
    printf("a=%d b=%.1f\n", out_a, out_b);
    printf("sizeof=%zu alignof=%zu\n", sizeof(Probe), alignof(Probe));
    return 0;
}
