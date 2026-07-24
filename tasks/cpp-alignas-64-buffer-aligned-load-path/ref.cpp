#include "sol.hpp"
#include <cstring>

uint64_t fill_aligned_buffer(unsigned char* storage, uint64_t base_address,
                              const float* data, int n) {
    const uint64_t ALIGN = 64;
    uint64_t aligned = (base_address + ALIGN - 1) & ~(ALIGN - 1);
    uint64_t offset = aligned - base_address;
    std::memcpy(storage + offset, data, sizeof(float) * (size_t)n);
    return aligned;
}
