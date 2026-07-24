#pragma once

// N thread-local variables, each `var_bytes` bytes, are laid out
// back-to-back starting at a cache-line-aligned base address. Because
// they're contiguous, several of them can land in the same `line_bytes`
// -byte cache line -- so whichever threads own those variables ping-pong
// the line back and forth (false sharing) even though they never touch
// each other's data. The fix: pad every variable up to a whole number of
// lines so each one gets a dedicated stretch of line(s) all to itself.
struct PadResult {
    long padding_bytes;  // bytes appended after ONE var so it fills whole lines
    long stride_bytes;   // var_bytes + padding_bytes: start-to-start distance
    long total_bytes;    // stride_bytes * n: size of the padded array of all n vars
};

// padding_bytes = (line_bytes - (var_bytes mod line_bytes)) mod line_bytes
// stride_bytes  = var_bytes + padding_bytes
// total_bytes   = stride_bytes * n
//
// The outer `mod line_bytes` matters: a var that ALREADY occupies a whole
// number of lines (var_bytes mod line_bytes == 0) needs ZERO padding, not
// a full extra line's worth.
PadResult min_padding_for_n_vars(long var_bytes, int line_bytes, long n);
