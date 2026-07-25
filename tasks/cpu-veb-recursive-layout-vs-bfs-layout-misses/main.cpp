#include <cstdio>
#include <set>
#include "sol.hpp"

// HARNESS baseline (not learner code): the standard level-order/heap
// array position -- root at 0, left child of p at 2p+1, right at 2p+2.
static int bfs_pos(const bool* path, int depth) {
    int idx = 0;
    for (int k = 0; k < depth; k++) idx = 2 * idx + (path[k] ? 2 : 1);
    return idx;
}

// Walks a root-to-leaf search (every prefix of `leaf_path`, depth 0
// through H-1) and counts the DISTINCT line_bytes-byte lines it touches
// under each layout.
static void search_counts(const bool* leaf_path, int H, int elem_bytes, int line_bytes,
                           long& veb_lines, long& bfs_lines) {
    std::set<long> vset, bset;
    for (int depth = 0; depth < H; depth++) {
        int vp = veb_pos(leaf_path, depth, H);
        int bp = bfs_pos(leaf_path, depth);
        vset.insert(((long)vp * elem_bytes) / line_bytes);
        bset.insert(((long)bp * elem_bytes) / line_bytes);
    }
    veb_lines = (long)vset.size();
    bfs_lines = (long)bset.size();
}

// FIXED driver. A complete binary tree of height H=8 (255 nodes, 8-byte
// records, 64-byte lines -- 8 nodes/line), searched along 3 fixed
// root-to-leaf paths.
int main() {
    const int H = 8, ELEM = 8, LINE = 64;

    const bool leftmost[7] = {false, false, false, false, false, false, false};
    const bool rightmost[7] = {true, true, true, true, true, true, true};
    const bool zigzag[7] = {false, true, false, true, false, true, false};

    long v1, b1, v2, b2, v3, b3;
    search_counts(leftmost, H, ELEM, LINE, v1, b1);
    search_counts(rightmost, H, ELEM, LINE, v2, b2);
    search_counts(zigzag, H, ELEM, LINE, v3, b3);

    printf("leftmost: veb=%ld bfs=%ld | rightmost: veb=%ld bfs=%ld | zigzag: veb=%ld bfs=%ld\n",
           v1, b1, v2, b2, v3, b3);
    return 0;
}
