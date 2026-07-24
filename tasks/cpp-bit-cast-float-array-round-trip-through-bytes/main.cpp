#include <cstdio>
#include <cstdint>
#include <cstring>
#include "sol.hpp"

// FIXED driver. A deterministic float fixture (positive/negative, integral
// and fractional magnitudes, and the +0.0/-0.0 pair whose sign bit only a
// bit-exact round trip preserves), serialized then deserialized. Both
// buffers are poisoned first so a no-op implementation prints poison, not
// coincidentally-correct zeros.
int main() {
    const int N = 8;
    static const float x[N] = {
        1.5f, -2.25f, 0.0f, -0.0f, 3.0f, -123456.0f, 0.000244140625f, 0.125f
    };

    unsigned char buf[4 * N];
    for (int i = 0; i < (int)sizeof(buf); i++) buf[i] = 0xCC;  // poison

    floats_to_bytes(x, N, buf);
    for (int i = 0; i < (int)sizeof(buf); i++) printf("%d ", (int)buf[i]);
    printf("\n");

    float y[N];
    for (int i = 0; i < N; i++) y[i] = -999.0f;  // poison

    bytes_to_floats(buf, N, y);

    // Print the recovered bit pattern (not the float value): printf("%f",
    // -0.0f) hides the sign bit, so a value-only comparison would miss a
    // broken -0.0 round trip.
    for (int i = 0; i < N; i++) {
        uint32_t bits;
        std::memcpy(&bits, &y[i], 4);
        printf("%u ", bits);
    }
    printf("\n");
    return 0;
}
