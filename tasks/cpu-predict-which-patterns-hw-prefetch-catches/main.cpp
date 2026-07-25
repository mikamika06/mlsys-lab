#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver: 5 deterministic byte-address traces, 4-byte (int32)
// elements, one page = 4096 bytes.
//   pattern 0: sequential,        256 elements, stride  4 B
//   pattern 1: fixed stride,       64 elements, stride 16 B
//   pattern 2: "random",          256 elements, deterministic
//              multiplicative permutation (irregular stride)
//   pattern 3: "pointer chase",   256 elements, a DIFFERENT
//              deterministic permutation (irregular stride)
//   pattern 4: large stride,       64 elements, stride 4096 B (1 page)
int main() {
    const long PAGE = 4096;
    std::vector<long> seq(256), stride16(64), rnd(256), chase(256), big(64);
    for (int i = 0; i < 256; i++) seq[i] = (long)i * 4;
    for (int i = 0; i < 64; i++) stride16[i] = (long)i * 16;
    for (int i = 0; i < 256; i++) rnd[i] = (long)((i * 167 + 13) % 256) * 4;
    for (int i = 0; i < 256; i++) chase[i] = (long)((i * 97 + 55) % 256) * 4;
    for (int i = 0; i < 64; i++) big[i] = (long)i * PAGE;

    const long* addrs[5] = { seq.data(), stride16.data(), rnd.data(), chase.data(), big.data() };
    int lens[5] = { (int)seq.size(), (int)stride16.size(), (int)rnd.size(), (int)chase.size(), (int)big.size() };
    int out[5] = {0, 0, 0, 0, 0};

    classify_prefetch(addrs, lens, 5, out);

    for (int k = 0; k < 5; k++) printf("pattern%d=%d\n", k, out[k]);
    return 0;
}
