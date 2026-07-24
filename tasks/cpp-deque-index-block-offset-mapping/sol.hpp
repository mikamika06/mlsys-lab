#pragma once
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// std::deque-style block-map arithmetic. A real deque stores its elements
// in a sequence of fixed 512-byte blocks; the number of elements that fit
// in one block is
//     N = max(1, 512 / elem_size)
// (integer division). Because a deque can grow at the front as well as the
// back, logical element 0 is not necessarily block-aligned: it lives at
// `first_offset` slots into block 0.
//
// For each logical index i in `indices` (0-based, i=0 is the current
// front element), compute
//     absolute      = first_offset + i
//     block_index   = absolute / N
//     block_offset  = absolute % N
// and append {block_index, block_offset} to the result, in the same order
// as `indices`.
// ---------------------------------------------------------------------------
std::vector<std::pair<long, long>> deque_mapping(long elem_size, long first_offset,
                                                   const std::vector<long>& indices);
