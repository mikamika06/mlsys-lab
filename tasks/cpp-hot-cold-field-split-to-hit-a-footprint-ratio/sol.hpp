#pragma once

// Footprint calculator, DEFINED in main.cpp (harness-side, not your job):
// applies the standard struct-layout rule for fundamental types -- a field
// of byte-size s must start at the next offset that is a multiple of s
// (natural alignment == size for char/short/int/double/pointer), and the
// struct's total size is then rounded up to a multiple of the LARGEST field
// size present, so the struct self-aligns correctly inside an array.
int struct_size(const int* sizes, int n);

// `fields[0..n)` are a fat struct's field byte-sizes, in their ORIGINAL
// declaration order. `is_hot[0..n)` (0 or 1) marks which fields are "hot"
// (touched every frame, e.g. in a tight per-entity loop) vs "cold" (rarely
// touched, e.g. debug/editor-only data).
//
// Fill hot_out[0 .. hot_count) with exactly the hot fields' sizes (the same
// multiset as fields[i] for every i with is_hot[i] != 0), REORDERED to make
// struct_size(hot_out, hot_count) as SMALL as possible -- that's the whole
// point of splitting: the tight loop's struct should be as compact as
// padding allows.
// Fill cold_out[0 .. n - hot_count) with exactly the cold fields' sizes, in
// their ORIGINAL relative order (the cold struct's padding is not graded --
// it's touched rarely, so its footprint doesn't matter here).
// Return hot_count, the number of hot fields.
int split_struct(const int* fields, const int* is_hot, int n,
                  int* hot_out, int* cold_out);
