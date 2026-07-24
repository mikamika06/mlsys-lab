#pragma once
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// PROVIDED (do not change): the real, compiler-laid-out quantization block.
// Under the LP64 ABI, `zero_point` (4-byte aligned) forces 3 bytes of
// padding after the 1-byte `scale`, so the record is NOT simply "1 + 4 + 16"
// bytes back to back -- its true size and field offsets come from whatever
// this actual compiler decides for this actual struct.
// ---------------------------------------------------------------------------
struct QuantBlock {
    int8_t  scale;        // 8-bit signed scale
    int32_t zero_point;   // 32-bit signed zero point (4-byte aligned)
    uint8_t weights[16];  // 32 packed int4 weights, 2 per byte
};

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Serialize a QuantBlock into `out` (a caller-owned buffer of exactly
// `out_len == sizeof(QuantBlock)` bytes), matching the REAL memory layout
// the compiler gives QuantBlock on this machine byte-for-byte:
//
//   1. Pack the 32 entries of `weights` (each in [0, 15]) into 16 bytes:
//      for i in [0,16), weights[2*i] goes in the LOW nibble and
//      weights[2*i+1] goes in the HIGH nibble of packed byte i.
//   2. Write `scale`, `zero_point`, then the 16 packed bytes into `out` at
//      QuantBlock's real field offsets (including any compiler-inserted
//      padding between fields), little-endian.
//
// A correct implementation is exactly what you get by writing the fields
// into an actual `QuantBlock` and copying its raw bytes -- that IS the
// ground truth here, there is no separate "spec" to hand-compute against.
// ---------------------------------------------------------------------------
void pack_quant_block(int8_t scale, int32_t zero_point, const int weights[32],
                       uint8_t* out, int out_len);
