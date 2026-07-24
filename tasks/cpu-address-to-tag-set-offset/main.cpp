#include "sol.hpp"
#include <cstdio>

struct Config { int line_bytes; int sets; int ways; };

int main() {
    static const Config configs[] = {
        {64, 64, 8},   // 32 KiB, 8-way, 64B lines
        {32, 128, 4},  // 16 KiB, 4-way, 32B lines
        {64, 1, 16},   // fully-associative-like: 1 set
        {16, 256, 1},  // direct-mapped, 16B lines
    };
    static const unsigned long addrs[] = {
        0, 15, 16, 63, 64, 65, 1000, 4096, 65536,
        123456, 1000000, 4294967295UL, 8589934592UL,
    };

    for (const auto& cfg : configs) {
        for (unsigned long a : addrs) {
            AddrDecomp d = decompose_address(a, cfg.line_bytes, cfg.sets, cfg.ways);
            printf("%lu %lu %lu\n", d.tag, d.set_index, d.offset);
        }
    }
    return 0;
}
