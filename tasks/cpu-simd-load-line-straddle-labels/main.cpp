#include <cstdint>
#include <cstdio>
#include "sol.hpp"

// FIXED driver. Every (base_addr, width_bytes) pair is hand-picked (no
// rand()/time()), fixed 64-byte cache lines, covering scalar (8), SSE (16),
// AVX (32) and AVX-512 (64) load widths at aligned, mid-line, and
// worst-case near-boundary base addresses.

namespace {
void run(const char* name, uint64_t base_addr, int width_bytes, int line_bytes) {
    bool s = straddles_line(base_addr, width_bytes, line_bytes);
    printf("%s base=%llu width=%d straddle=%d\n", name,
           static_cast<unsigned long long>(base_addr), width_bytes, s ? 1 : 0);
}
}  // namespace

int main() {
    constexpr int L = 64;
    constexpr uint64_t kBase = 0x10000;  // arbitrary line-aligned region base

    run("scalar_aligned", kBase, 8, L);            // offset 0,  0+8=8   <=64  -> no
    run("zmm_full_line_aligned", kBase, 64, L);     // offset 0,  0+64=64 <=64  -> no
    run("xmm_ends_exactly_on_boundary", kBase + 48, 16, L);  // offset 48, 48+16=64 -> no
    run("ymm_crosses_boundary", kBase + 48, 32, L);          // offset 48, 48+32=80 -> yes
    run("zmm_mid_line", kBase + 32, 64, L);                  // offset 32, 32+64=96 -> yes
    run("scalar_worst_case", kBase + 63, 8, L);               // offset 63, 63+8=71  -> yes
    run("zmm_off_by_one", kBase + 1, 64, L);                  // offset 1,  1+64=65  -> yes
    run("xmm_mid_line_fits", kBase + 16, 16, L);               // offset 16, 16+16=32 -> no
    return 0;
}
