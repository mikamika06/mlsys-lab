#include <cstdint>
#include <cstdio>
#include <list>
#include <unordered_map>
#include <vector>
#include "sol.hpp"

// FIXED driver + FIXED two-level cache model. Deterministic A/B contents
// (no rand()/time()). M=N=K=64, but leading dimensions are padded to 65
// (deliberately NOT a multiple of either level's set-count*line-size), so
// this task measures tiling, not leading-dimension aliasing.

namespace {

constexpr int kLineBytes = 64;

struct SetAssocLru {
    int sets, ways;
    std::vector<std::list<uint64_t>> lists;
    std::vector<std::unordered_map<uint64_t, std::list<uint64_t>::iterator>> pos;

    SetAssocLru(int s, int w) : sets(s), ways(w), lists(s), pos(s) {}

    void reset() {
        for (auto& l : lists) l.clear();
        for (auto& p : pos) p.clear();
    }

    // Returns true on hit. Always inserts the line on a miss.
    bool access(uint64_t line) {
        int set_idx = static_cast<int>(line % static_cast<uint64_t>(sets));
        auto& lst = lists[set_idx];
        auto& mp = pos[set_idx];
        auto it = mp.find(line);
        if (it != mp.end()) {
            lst.erase(it->second);
            lst.push_front(line);
            mp[line] = lst.begin();
            return true;
        }
        if (static_cast<int>(lst.size()) >= ways) {
            uint64_t victim = lst.back();
            lst.pop_back();
            mp.erase(victim);
        }
        lst.push_front(line);
        mp[line] = lst.begin();
        return false;
    }
};

SetAssocLru g_l1(8, 2);
SetAssocLru g_l2(64, 8);
MissVector g_misses{0, 0};

double a_value(int i, int k) {
    return static_cast<double>((i * 131 + k * 977) % 1009) * 0.01;
}
double b_value(int k, int j) {
    return static_cast<double>((k * 733 + j * 331) % 907) * 0.01;
}

}  // namespace

void cache_reset() {
    g_l1.reset();
    g_l2.reset();
    g_misses = MissVector{0, 0};
}

void touch(const void* p) {
    uint64_t line = reinterpret_cast<uint64_t>(p) / static_cast<uint64_t>(kLineBytes);
    bool l1_hit = g_l1.access(line);
    if (l1_hit) return;
    ++g_misses.l1_misses;
    bool l2_hit = g_l2.access(line);
    if (!l2_hit) ++g_misses.l2_misses;
}

MissVector miss_vector() { return g_misses; }

int main() {
    constexpr int M = 64, N = 64, K = 64;
    constexpr int lda = 65, ldb = 65, ldc = 65;

    std::vector<double> A(static_cast<size_t>(M) * lda, 0.0);
    std::vector<double> B(static_cast<size_t>(K) * ldb, 0.0);
    std::vector<double> C(static_cast<size_t>(M) * ldc, 0.0);

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A[static_cast<size_t>(i) * lda + k] = a_value(i, k);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            B[static_cast<size_t>(k) * ldb + j] = b_value(k, j);

    cache_reset();
    matmul_two_level_tiled(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    MissVector m = miss_vector();

    double checksum = 0.0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            checksum += C[static_cast<size_t>(i) * ldc + j];

    printf("M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n", M, N, K, lda, ldb, ldc);
    printf("checksum=%.4f\n", checksum);
    printf("l1_misses=%d l2_misses=%d\n", m.l1_misses, m.l2_misses);
    return 0;
}
