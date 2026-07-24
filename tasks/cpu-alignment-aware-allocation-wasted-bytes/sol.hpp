#pragma once
// Simulate a bump allocator: starting at offset 0, for each of n requests
// (sizes[i], alignments[i]) in order, round the CURRENT offset UP to the
// next multiple of alignments[i] -- that gap is padding wasted purely to
// satisfy alignment -- then advance the offset by that padding plus
// sizes[i]. Every alignments[i] is a power of two (4, 8, 16, 32, or 64).
//
// Return the TOTAL padding bytes summed across all n requests. Never count
// the requested sizes themselves as "wasted".
long total_wasted_bytes(const int* sizes, const int* alignments, int n);
