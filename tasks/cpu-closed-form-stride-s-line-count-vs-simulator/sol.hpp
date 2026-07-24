#pragma once

// A stride walk reads n elements of `elem_bytes` bytes each, at byte
// addresses 0, stride*elem_bytes, 2*stride*elem_bytes, ...,
// (n-1)*stride*elem_bytes. Grouping those addresses into `line_bytes`-byte
// cache lines (line k covers bytes [k*line_bytes, (k+1)*line_bytes)),
// return the number of DISTINCT lines touched.
//
// Derive this algebraically from n, stride, elem_bytes, line_bytes --
// no per-element simulation loop. In every scenario this task uses,
// line_bytes is an exact multiple of elem_bytes.
long distinct_lines_stride_walk(long n, long stride, long elem_bytes, long line_bytes);
