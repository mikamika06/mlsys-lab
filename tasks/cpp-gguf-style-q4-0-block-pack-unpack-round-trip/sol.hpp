#pragma once
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// PROVIDED (do not change): the real GGUF Q4_0 block layout. Under this
// compiler's LP64 ABI it packs to exactly 18 bytes with no padding
// (uint16_t needs only 2-byte alignment, uint8_t[16] needs 1).
// ---------------------------------------------------------------------------
struct block_q4_0 {
    uint16_t d;       // IEEE-754 binary16 scale, raw bits
    uint8_t  qs[16];  // 32 packed int4 weights, 2 per byte
};

// PROVIDED (defined in main.cpp): IEEE-754 binary16 encode/decode,
// round-to-nearest-even. Use these for the scale -- do not hand-roll your
// own float16 bit twiddling, that is not what this task is about.
uint16_t encode_fp16(float x);
float    decode_fp16(uint16_t h);

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS both.
//
// pack_q4_0: pack 32 float32 weights into an 18-byte block_q4_0, writing
// its raw bytes into `out` (out_len == sizeof(block_q4_0)):
//   1. d = max_i(|weights[i]|) / 7.0f   (if every weight is 0, d = 0)
//   2. for i in [0,32): q[i] = round(weights[i] / d) clamped to [-8, 7]
//      (if d == 0, every q[i] = 0 -- do not divide by zero)
//   3. store (q[i] + 8) in the LOW nibble of qs[i] and (q[i+16] + 8) in
//      the HIGH nibble of qs[i], for i in [0, 16)
//   4. encode d with encode_fp16() into the block's `d` field
//
// unpack_q4_0: the inverse. Read an 18-byte block_q4_0 from `block`
// (block_len == sizeof(block_q4_0)) and reconstruct 32 approximate
// weights into out_weights[32]:
//   d = decode_fp16(block.d)
//   for i in [0,16): q0 = (qs[i] & 0xF) - 8, q1 = (qs[i] >> 4) - 8
//                    out_weights[i] = q0 * d, out_weights[i+16] = q1 * d
// ---------------------------------------------------------------------------
void pack_q4_0(const float weights[32], uint8_t* out, int out_len);
void unpack_q4_0(const uint8_t* block, int block_len, float out_weights[32]);
