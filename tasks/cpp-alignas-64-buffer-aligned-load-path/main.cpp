#include <cstdio>
#include <cstdint>
#include <cstring>
#include "sol.hpp"

// FIXED driver. `storage` is a real alignas(64) buffer; `base_address` is a
// synthetic (not-really-mapped) address whose low bits force a nonzero skip
// before the next 64-byte boundary. Everything printed is bit-exact and
// independent of ASLR / real pointer values, since we never print a real
// address — only the synthetic `aligned` value the candidate computes and
// the bytes actually landed in `storage`.
int main() {
    const int BUF_SIZE = 256;
    alignas(64) static unsigned char storage[BUF_SIZE];
    std::memset(storage, 0, BUF_SIZE);

    const uint64_t base_address = 1000;   // 1000 % 64 == 40, not aligned
    const float data[5] = {3.14f, 2.71f, -1.5f, 0.0f, 42.125f};
    const int n = 5;

    uint64_t aligned = fill_aligned_buffer(storage, base_address, data, n);

    printf("%llu\n", (unsigned long long)aligned);
    printf("%llu\n", (unsigned long long)(aligned % 64));
    for (int i = 0; i < BUF_SIZE; i++) {
        printf("%d\n", (int)storage[i]);
    }
    return 0;
}
