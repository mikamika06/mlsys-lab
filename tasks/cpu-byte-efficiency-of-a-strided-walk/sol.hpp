#pragma once

// Harness hooks (declared here, DEFINED in main.cpp).
void reset_touch();                 // clear the set of touched cache lines
void touch(long byte_addr);         // mark one byte address as touched (buckets into a 64-byte line)
long touched_line_count();          // distinct 64-byte lines touched since the last reset_touch()

// Walk n elements of `width` bytes each, starting at byte 0, with a
// stride of `stride_elems` elements between consecutive accessed
// elements (so element i starts at byte i * stride_elems * width). Call
// touch(addr) once for every byte address in
//   [i * stride_elems * width, i * stride_elems * width + width)
// for every i in [0, n).
void walk(int n, int stride_elems, int width);

// Return bytes_used / bytes_fetched for this walk:
//   bytes_used    = n * width
//   bytes_fetched = touched_line_count() * 64, measured by calling
//                   reset_touch() then walk(n, stride_elems, width).
double byte_efficiency(int n, int stride_elems, int width);
