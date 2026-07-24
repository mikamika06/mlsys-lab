#include "sol.hpp"

// TODO: copy each field from the AoS record into its own SoA array.
void aos_to_soa(const Particle* aos, int n, float* xs, float* ys, float* zs, int* ids) {
    (void)aos;
    (void)n;
    (void)xs;
    (void)ys;
    (void)zs;
    (void)ids;
    // your code here
}

// TODO: copy each SoA array's i-th element back into the i-th AoS record.
void soa_to_aos(const float* xs, const float* ys, const float* zs, const int* ids, int n,
                 Particle* aos) {
    (void)xs;
    (void)ys;
    (void)zs;
    (void)ids;
    (void)n;
    (void)aos;
    // your code here
}
