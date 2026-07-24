#include <cstdint>
#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Deterministic, no rand()/time(): a splitmix64-style LCG with
// a fixed seed generates every input array. Prints every output element
// (plus a checksum) so the entire behaviour is visible in stdout and
// compared byte-for-byte against the reference build.

namespace {

struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // uniform int in [lo, hi] inclusive
    int32_t range(int32_t lo, int32_t hi) {
        uint64_t span = static_cast<uint64_t>(hi - lo) + 1;
        return lo + static_cast<int32_t>(next() % span);
    }
};

}  // namespace

int main() {
    constexpr int N = 24;
    Lcg rng(20260724ULL);

    // -- select_branchless: cond in {0,1}, a/b spanning negative & positive.
    std::vector<int32_t> cond(N), a(N), b(N), sel_out(N);
    for (int i = 0; i < N; ++i) {
        cond[i] = rng.range(0, 1);
        a[i] = rng.range(-1000, 1000);
        b[i] = rng.range(-1000, 1000);
    }
    select_branchless(cond.data(), a.data(), b.data(), sel_out.data(), N);

    long long sel_checksum = 0;
    for (int i = 0; i < N; ++i) {
        printf("sel %d %d %d %d -> %d\n", i, cond[i], a[i], b[i], sel_out[i]);
        sel_checksum += sel_out[i];
    }
    printf("sel_checksum=%lld\n", sel_checksum);

    // -- clamp_branchless: x spanning well outside [lo, hi] on both sides.
    std::vector<int32_t> x(N), clamp_out(N);
    for (int i = 0; i < N; ++i) x[i] = rng.range(-1500, 1500);
    const int32_t lo = -200, hi = 300;
    clamp_branchless(x.data(), lo, hi, clamp_out.data(), N);

    long long clamp_checksum = 0;
    for (int i = 0; i < N; ++i) {
        printf("clamp %d %d -> %d\n", i, x[i], clamp_out[i]);
        clamp_checksum += clamp_out[i];
    }
    printf("clamp_checksum=%lld\n", clamp_checksum);

    return 0;
}
