#pragma once
#include <cstdint>

// Write `n` floats into a 64-byte-aligned load path inside `storage`.
//
// `storage` is a real buffer that the driver has already declared
// `alignas(64)`, so storage[0] itself sits on a 64-byte boundary. You are
// handed a hypothetical raw memory address `base_address` (NOT the real
// address of `storage` — a synthetic value modelling "where a real
// allocator happened to hand you memory", whose low bits are not a
// multiple of 64). Your job:
//
//   1. Compute the first address `aligned` with `aligned >= base_address`
//      and `aligned % 64 == 0`, using
//         aligned = (base_address + 64 - 1) & ~(64 - 1)
//   2. That address is `(aligned - base_address)` bytes further into the
//      buffer than `base_address` would be, i.e. offset = aligned - base_address.
//      Because storage[0] is itself 64-aligned, `storage + offset` is the
//      real, in-buffer location that lands on the same 64-byte boundary.
//   3. Copy the `n` floats from `data` into `storage[offset .. offset +
//      4*n)` as raw bytes (native little-endian layout — a plain float
//      write/memcpy is exactly this).
//   4. Return `aligned`.
//
// Every other byte of `storage` must stay untouched (it is zero-initialised
// by the driver before the call).
uint64_t fill_aligned_buffer(unsigned char* storage, uint64_t base_address,
                              const float* data, int n);
