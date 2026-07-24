#include "sol.hpp"

// Correct AoS sweep: reading arr[i].x still pulls in the whole record, so
// each element visited is one touch() on the record's address.
float sum_field_aos(const Particle* arr, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        touch(&arr[i]);
        s += arr[i].x;
    }
    return s;
}

// Correct SoA sweep: each element visited is one touch() on the x-array
// address — nothing else is fetched.
float sum_field_soa(const float* xs, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        touch(&xs[i]);
        s += xs[i];
    }
    return s;
}
