#pragma once

// A complete binary tree of height H (H levels, 2^H - 1 nodes, leaves
// at depth H-1) is laid out in an array by the van Emde Boas (vEB)
// recursive layout. A node is identified by `path[0..depth)`, its
// sequence of left(false)/right(true) choices from the root (`depth`
// counts levels below the root; `depth == 0` means the root itself).
//
// The layout is defined recursively on a SUBTREE's own height h
// (starting at h = H for the whole tree, occupying array slots
// [0, 2^H - 1)): split h into a TOP half h1 = ceil(h/2) and, for each
// of the top subtree's 2^h1 leaves, a BOTTOM subtree of height
// h2 = h - h1. The top subtree occupies the FIRST 2^h1 - 1 slots of
// this subtree's block, laid out the same way recursively. Then each
// bottom subtree occupies the next 2^h2 - 1 slots, one block per
// top-subtree leaf, in left-to-right leaf order, each laid out the same
// way recursively.
//
// Return the node's final flat array slot, an integer in [0, 2^H - 1).
int veb_pos(const bool* path, int depth, int H);
