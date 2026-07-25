#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache (64-byte lines, 16 sets, 4-way -- 4096
// bytes / 1024 floats total capacity). Call touch() once per DATA
// element you read, at that element's BYTE ADDRESS (index * 4).
void touch(long byte_addr);

// `data[0..dsize)` is far bigger than the modelled cache. `idx[0..n)`
// are indices into `data` (0 <= idx[i] < dsize); the SAME index value
// commonly appears more than once in `idx`, but its occurrences can be
// far apart in `i`-order.
//
// Compute out[i] = data[idx[i]] for every i in [0, n). Call
// touch((long)idx[i] * sizeof(float)) EXACTLY ONCE per i (n touches
// total) -- but you may issue those touches, and do the corresponding
// reads, in ANY ORDER (segment/sort the request stream by target index
// so repeats of the same index land close together and reuse the line
// instead of being re-fetched). Regardless of processing order, `out`
// must end up byte-identical to filling it in `i`-order: out[i] is
// always data[idx[i]].
void segmented_gather(const float* data, int dsize, const int* idx, int n, float* out);
