#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// A paged KV-cache allocator rounds every token's `s`-byte KV entry up to a
// whole number of `b`-byte blocks: `ceil(s / b)` blocks, costing
// `ceil(s / b) * b` bytes of storage PLUS `table_overhead_per_block` bytes
// of block-table metadata for EACH of those blocks.
//
// Candidate block sizes are the powers of two `{16, 32, 64, 128, 256, 512,
// 1024}`. For a workload of `n` tokens `token_sizes[0..n)`, for a given
// candidate `b`:
//
//   allocated(b) = sum_i( ceil(token_sizes[i] / b) * b )
//                  + total_blocks(b) * table_overhead_per_block
//   total_blocks(b) = sum_i( ceil(token_sizes[i] / b) )
//
// The useful bytes (plain sum of token_sizes) is the SAME for every
// candidate, so the block size that maximizes useful/allocated is exactly
// the one that MINIMIZES allocated(b) -- no floating point needed to
// choose it.
//
// Write the chosen block size to *out_block_size, the (candidate-
// independent) useful byte total to *out_useful_bytes, and that chosen
// candidate's allocated(b) to *out_allocated_bytes. Break ties between
// equally-good candidates by picking the SMALLER block size.
// ============================================================================
void choose_kv_block_size(const int* token_sizes, int n, int table_overhead_per_block,
                           int* out_block_size, long* out_useful_bytes, long* out_allocated_bytes);
