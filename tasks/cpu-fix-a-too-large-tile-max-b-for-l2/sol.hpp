#pragma once

// A blocked (tiled) matmul kernel's inner loop keeps THREE B x B tiles of
// 4-byte floats resident at once -- a tile of A, a tile of B, and the C
// accumulator tile -- so they all stay in L2 while the inner loop reuses
// them repeatedly. Given the L2 capacity in bytes, return the LARGEST
// integer B such that all three tiles fit together:
//
//   3 * B * B * 4 <= l2_bytes
//
// i.e. the largest B for which (B+1) would no longer fit.
int max_tile_b_for_l2(long l2_bytes);
