#pragma once
#include <cstddef>
#include <cstdint>

// ============================================================================
// Fixed record types (real C++ — sizes/alignment come from the actual
// compiler, not a modeled ABI). Particle is the Array-of-Structs record:
// three floats + an int, 16 bytes under the platform's natural alignment
// (no padding is needed since every member is 4 bytes). Its Struct-of-Arrays
// counterpart replaces the array of Particle with one contiguous array per
// field.
// ============================================================================
struct Particle {
    float x, y, z;
    int id;
};

// ============================================================================
// Fake cache-line probe (FIXED — do not modify; defined in main.cpp).
// touch(p) records which 64-byte cache line the address `p` falls in, into a
// set that cache_reset() clears. lines_touched() returns how many DISTINCT
// lines have been recorded since the last reset. This is a deterministic
// stand-in for hardware cache misses (which are not reproducible): the driver
// 64-byte-aligns every array it hands you, so the line count depends only on
// which addresses your code actually visits.
// ============================================================================
void cache_reset();
void touch(const void* p);
int lines_touched();

// ============================================================================
// LEARNER implements these two sweeps in solve.cpp.
//
// sum_field_aos: sum the `x` field over an Array-of-Structs of `n` Particles.
// For every element you read, call touch(&arr[i]) exactly once (accessing
// any field of arr[i] pulls in the whole 16-byte record's cache line — the
// other three fields are wasted bandwidth).
//
// sum_field_soa: sum an SoA `x`-field array of `n` floats directly. For
// every element you read, call touch(&xs[i]) exactly once — no other field
// exists in this array, so nothing wasted is fetched.
//
// A correct AoS sweep of N=1024 elements (16 B/record) touches 256 distinct
// 64-byte lines; the equivalent SoA sweep (4 B/record) touches 64 — exactly
// the sizeof(field)/sizeof(struct) = 4/16 = 0.25 ratio the task is about.
// ============================================================================
float sum_field_aos(const Particle* arr, int n);
float sum_field_soa(const float* xs, int n);
