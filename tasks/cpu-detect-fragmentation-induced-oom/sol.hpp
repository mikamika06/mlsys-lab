#pragma once

// A heap is a fixed list of `num_blocks` blocks with byte sizes
// `block_sizes[0..num_blocks)`. Every block starts FREE. Block
// boundaries never change -- no splitting, no coalescing -- a block only
// toggles between FREE and USED.
//
// `num_ops` operations then run in order, op i described by
// `op_types[i]`:
//   0 = ALLOC: request `op_sizes[i]` bytes. Using FIRST-FIT (scan blocks
//       in index order), find the first FREE block whose size is
//       >= the request. If found, mark it USED and set
//       `out_labels[i] = 1` (succeed). If no free block is big enough --
//       even if the FREE blocks' sizes sum to more than the request --
//       the allocation FAILS: set `out_labels[i] = 0`, heap state
//       unchanged. That "sum is enough, no single block is" case is
//       external fragmentation.
//   1 = FREE: `op_ids[i]` holds the op-index of the ALLOC being freed
//       (guaranteed to be a prior op that succeeded). Mark the block
//       that ALLOC used FREE again. `out_labels[i]` is not read by the
//       driver for FREE ops; leave it whatever it already is.
//
// `out_labels` has length `num_ops` and starts zero-filled.
void classify_allocations(const int* block_sizes, int num_blocks,
                           const int* op_types, const int* op_sizes, const int* op_ids,
                           int num_ops, int* out_labels);
