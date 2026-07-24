#pragma once

struct DataChunk {
    int header;
    double values[4];
};

// Populate the first `num_chunks` elements of `chunks`:
//     chunks[i].header    = i
//     chunks[i].values[j] = i * 1.5 + j        for j in [0, 4)
//
// `chunks` was heap-allocated with exactly `num_chunks` logical elements
// PLUS one hidden GUARD chunk right after it, at chunks[num_chunks], that
// the caller uses to detect an out-of-bounds write. Touch ONLY indices
// [0, num_chunks) — an off-by-one loop that also writes chunks[num_chunks]
// is a heap buffer overflow (the classic bug AddressSanitizer is built to
// catch) and will corrupt the guard chunk the caller checks afterwards.
void populate_chunks(DataChunk* chunks, int num_chunks);
