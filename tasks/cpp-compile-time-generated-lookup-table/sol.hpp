#pragma once
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// PROVIDED (do not change): the real, compiler-laid-out LUT entry. This is
// exactly the kind of struct a `constexpr std::array<LutEntry, N>` table
// generator would emit -- its size and field offsets, including any
// alignment padding, come from the REAL compiler, not a hand-computed spec.
// ---------------------------------------------------------------------------
struct LutEntry {
    char  index;     // i, as an unsigned byte
    short doubled;   // i * 2
    int   squared;   // i * i
};

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Serialize n back-to-back LutEntry records into `out` (a caller-owned
// buffer of exactly out_len == n * sizeof(LutEntry) bytes), matching the
// REAL memory layout of `std::array<LutEntry, n>` byte-for-byte:
//
//   for i in [0, n):
//     entry i .index   = (char)  i
//     entry i .doubled = (short)(i * 2)
//     entry i .squared = (int)  (i * i)
//
// including every compiler-inserted padding byte (which must be 0x00), at
// entry i's real offset i * sizeof(LutEntry) in `out`.
//
// The simplest correct strategy: build the values into an actual
// `LutEntry` and copy its raw bytes -- that IS the ground truth, there is
// no separate spec to hand-compute against.
// ---------------------------------------------------------------------------
void generate_lut_bytes(int n, uint8_t* out, int out_len);
