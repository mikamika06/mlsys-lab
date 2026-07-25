#pragma once

// A 2D array VIEW of R rows and C columns, holding elements of
// `elem_bytes` bytes, described by `row_stride` and `col_stride` in
// ELEMENTS (not bytes): element (i, j) sits at byte address
//
//   base + (i * row_stride + j * col_stride) * elem_bytes
//
// One formula describes a row-major array (row_stride=C, col_stride=1),
// a column-major array (row_stride=1, col_stride=R), or a
// transposed/sliced VIEW of either, just by choosing which stride is 1.
long element_addr(long base, int i, int j, long row_stride, long col_stride, int elem_bytes);

// Traverse every one of the view's R*C elements exactly once --
// row_major=true: i outer, j inner (row by row); false: j outer, i
// inner (column by column) -- tracking a single "currently open" line:
// the line of the immediately preceding access (none before the
// first). Every access whose line differs from the currently-open one
// is a LINE FETCH: increment the count and that access's line becomes
// the new currently-open one; an access landing in the SAME line as the
// one before it costs nothing extra. Return the total fetch count over
// the whole traversal.
long traversal_fetch_count(long base, int R, int C, long row_stride, long col_stride,
                            int elem_bytes, int line_bytes, bool row_major);
