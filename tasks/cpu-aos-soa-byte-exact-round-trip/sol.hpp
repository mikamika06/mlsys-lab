#pragma once
// Array-of-Structs record.
struct Particle {
    float x, y, z;
    int id;
};

// Convert an AoS array of n particles into Struct-of-Arrays form: the
// i-th particle's fields land at xs[i], ys[i], zs[i], ids[i].
void aos_to_soa(const Particle* aos, int n, float* xs, float* ys, float* zs, int* ids);

// Convert SoA arrays of n elements back into an AoS array. Must be the
// exact inverse of aos_to_soa: converting AoS -> SoA -> AoS must reproduce
// every field byte-for-byte.
void soa_to_aos(const float* xs, const float* ys, const float* zs, const int* ids, int n,
                 Particle* aos);
