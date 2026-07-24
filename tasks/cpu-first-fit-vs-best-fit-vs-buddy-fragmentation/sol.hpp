#pragma once

// The trace: for i in [0, num_ops), op_kind[i] is 0 (ALLOC) or 1 (FREE).
//   ALLOC: request op_arg[i] bytes.
//   FREE:  free the block that was returned by the op_arg[i]-th ALLOC op
//          seen so far in THIS trace (0-based, counting only ALLOC ops).
//          Each of the three allocators below tracks its OWN offsets for
//          its own allocs, so "the op_arg[i]-th alloc's block" generally
//          means a DIFFERENT byte offset in each of the three arenas.
//
// Replay the SAME fixed op trace through three independent allocators, all
// managing their own separate 256-byte arena, and write each one's final
// EXTERNAL FRAGMENTATION into out[0..3):
//
//   out[0]  Address-ordered FIRST-FIT: maintains a list of blocks in
//           address order, each either used or free. alloc(size) scans
//           from the lowest address and takes the FIRST free block whose
//           size >= the request; if the block is bigger than needed, it
//           is SPLIT into a used prefix of exactly `size` bytes and a
//           free remainder immediately after it. free(offset) marks that
//           block free again, then coalesces it with its immediate left
//           and/or right neighbor in the list if either is also free
//           (repeat until neither neighbor is free -- a 3-way merge is
//           possible in one free() call).
//
//   out[1]  Address-ordered BEST-FIT: identical bookkeeping and free()
//           coalescing to first-fit, but alloc(size) scans ALL free
//           blocks and takes the SMALLEST one whose size >= the request
//           (ties broken by lowest address, i.e. whichever is found
//           first in address order), then splits it the same way.
//
//   out[2]  BUDDY: a classic power-of-two buddy allocator, 5 levels
//           (block sizes 256, 128, 64, 32, 16 -- 16 is the minimum block
//           size). alloc(size) rounds up to the smallest power-of-two
//           block size that fits (at least 16), finds the smallest
//           already-available free block that is at least that big, and
//           splits it down level by level until a block of exactly the
//           right size exists. free(offset) returns the block to its
//           level's free list, then merges it with its BUDDY -- and only
//           its buddy, never an arbitrary free neighbor -- repeatedly, as
//           far up the levels as the buddy chain stays free.
//
// A failed alloc() (no free block big enough anywhere) returns -1 and
// changes nothing; free() on an offset that was never (or is no longer) a
// live allocation in that arena is a no-op.
//
// EXTERNAL FRAGMENTATION, for each arena, after the whole trace has run:
//     total_free_bytes - largest_contiguous_free_block_bytes
// i.e. how many free bytes exist that a request for the largest available
// block size could NOT actually use, because they're scattered across
// smaller, non-contiguous (or, for buddy, non-merge-eligible) pieces.
void fragmentation_after_trace(const int* op_kind, const int* op_arg, int num_ops, int* out);
