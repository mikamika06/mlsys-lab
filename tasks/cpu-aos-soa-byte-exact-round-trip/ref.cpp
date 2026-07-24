#include "sol.hpp"

void aos_to_soa(const Particle* aos, int n, float* xs, float* ys, float* zs, int* ids) {
    for (int i = 0; i < n; i++) {
        xs[i] = aos[i].x;
        ys[i] = aos[i].y;
        zs[i] = aos[i].z;
        ids[i] = aos[i].id;
    }
}

void soa_to_aos(const float* xs, const float* ys, const float* zs, const int* ids, int n,
                 Particle* aos) {
    for (int i = 0; i < n; i++) {
        aos[i].x = xs[i];
        aos[i].y = ys[i];
        aos[i].z = zs[i];
        aos[i].id = ids[i];
    }
}
