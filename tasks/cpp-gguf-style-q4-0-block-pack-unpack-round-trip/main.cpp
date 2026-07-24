#include <cstdio>
#include <cstring>
#include "sol.hpp"

// PROVIDED. IEEE-754 binary16 <-> float32, round-to-nearest-even.
uint16_t encode_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t abse = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (abse == 0xFFu) {  // inf / nan
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }

    int32_t exp = (int32_t)abse - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  // underflows to zero
        mant |= 0x800000u;                      // implicit leading 1
        int shift = 14 - exp;                   // >= 14
        uint32_t half_mant = mant >> shift;
        uint32_t remainder = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
    }

    uint32_t half_mant = mant >> 13;
    uint32_t remainder = mant & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x400u) {
            half_mant = 0;
            exp++;
            if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

float decode_fp16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t e = 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            uint32_t fexp = e - 15u + 127u;
            f = sign | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t fexp = exp - 15u + 127u;
        f = sign | (fexp << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// PROVIDED. Deterministic value generator (no rand(), no clock).
static float detval(int i) {
    unsigned x = (unsigned)(i * 2654435761u + 97u);
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    return (((float)(x % 20000) / 100.0f) - 100.0f) * 0.05f;  // roughly [-5,5)
}

// FIXED driver. Do not edit. Two fixed weight blocks (mixed-magnitude, and
// all-zero) exercise the general path and the d==0 edge case. For each,
// packs into a sentinel-filled buffer, prints the packed hex bytes, then
// unpacks and prints the 32 reconstructed floats.
int main() {
    float w1[32];
    for (int i = 0; i < 32; ++i) w1[i] = detval(i);

    float w2[32];
    for (int i = 0; i < 32; ++i) w2[i] = 0.0f;

    float* cases[2] = {w1, w2};
    for (int c = 0; c < 2; ++c) {
        uint8_t out[sizeof(block_q4_0)];
        for (size_t i = 0; i < sizeof(out); ++i) out[i] = 0xFF;

        pack_q4_0(cases[c], out, (int)sizeof(out));

        printf("block_size=%d bytes:", (int)sizeof(block_q4_0));
        for (size_t i = 0; i < sizeof(out); ++i) printf("%02x", out[i]);
        printf("\n");

        float rebuilt[32];
        unpack_q4_0(out, (int)sizeof(out), rebuilt);
        printf("unpacked:");
        for (int i = 0; i < 32; ++i) printf(" %.6f", rebuilt[i]);
        printf("\n");
    }
    return 0;
}
