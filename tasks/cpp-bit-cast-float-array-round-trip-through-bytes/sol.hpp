#pragma once

// Serialize n floats into a byte buffer WITHOUT type-punning UB: convert
// each float to its 4-byte IEEE-754 representation with std::bit_cast (never
// reinterpret_cast<unsigned char*> + dereference, never a union), then write
// those 4 bytes little-endian into out[4*i .. 4*i+3]. `out` must hold at
// least 4*n bytes.
void floats_to_bytes(const float* x, int n, unsigned char* out);

// Recover n floats from a byte buffer written by floats_to_bytes(): read
// each little-endian 4-byte group back into a std::uint32_t and use
// std::bit_cast to reconstruct the float from it without UB.
void bytes_to_floats(const unsigned char* in, int n, float* out);
