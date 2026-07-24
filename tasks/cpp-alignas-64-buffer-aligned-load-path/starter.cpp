#include "sol.hpp"

// TODO: implement.
//   - Round base_address up to the next 64-byte boundary -> aligned.
//   - offset = aligned - base_address.
//   - memcpy n floats from data into storage + offset.
//   - return aligned.
uint64_t fill_aligned_buffer(unsigned char* storage, uint64_t base_address,
                              const float* data, int n) {
    // your code here
    return 0;
}
