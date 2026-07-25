#include "sol.hpp"

namespace {
int pow2(int e) {
    int r = 1;
    for (int i = 0; i < e; i++) r *= 2;
    return r;
}

// Position of the node at path[off..off+depth) within a subtree of
// height h whose own block starts at array offset `base`.
int veb_rec(const bool* path, int off, int depth, int h, int base) {
    if (h <= 1) return base;  // single-node (sub)tree

    int h1 = (h + 1) / 2;  // ceil(h/2)
    int h2 = h - h1;
    int top_size = pow2(h1) - 1;

    if (depth < h1) {
        return veb_rec(path, off, depth, h1, base);
    }

    int leaf_index = 0;
    for (int k = 0; k < h1; k++) leaf_index = leaf_index * 2 + (path[off + k] ? 1 : 0);

    int bottom_size = pow2(h2) - 1;
    int bottom_base = base + top_size + leaf_index * bottom_size;
    return veb_rec(path, off + h1, depth - h1, h2, bottom_base);
}
}  // namespace

int veb_pos(const bool* path, int depth, int H) {
    return veb_rec(path, 0, depth, H, 0);
}
