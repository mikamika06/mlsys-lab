#pragma once

// A free-list allocator manages a `heap_bytes`-byte heap, starting as one
// free block [0, heap_bytes). `num_ops` operations run in order,
// `op_types[i]` one of:
//   0 = ALLOC: request `op_sizes[i]` bytes. Using FIRST-FIT (scan the
//       heap's blocks in address order), find the first FREE block whose
//       size is >= the request (guaranteed to exist in this task's
//       scenarios -- every ALLOC succeeds). If the block is an exact
//       match, mark it USED. If it's bigger, SPLIT it: carve a used
//       block of exactly the requested size off its low-address end, and
//       leave the remaining bytes as a new, smaller free block right
//       after it.
//   1 = FREE: `op_ids[i]` is the op-index of the ALLOC being released
//       (always a prior op that succeeded). Mark that block FREE again,
//       then COALESCE: if the physically adjacent block on either side
//       (immediately before or after it in address order) is also free,
//       merge them into one bigger free block (checking both sides, so
//       a free op can merge up to three blocks into one).
//
// After running every op, return the heap's EXTERNAL FRAGMENTATION
// RATIO: the fraction of free bytes that are NOT part of the single
// largest free block --
//
//   (total_free_bytes - largest_free_block_bytes) / total_free_bytes
//
// as a double (return 0.0 if total_free_bytes is 0).
double external_fragmentation_ratio(long heap_bytes,
                                     const int* op_types, const int* op_sizes, const int* op_ids,
                                     int num_ops);
