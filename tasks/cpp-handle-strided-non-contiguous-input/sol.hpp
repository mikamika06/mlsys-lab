#pragma once
#include <cstdint>

// Reduces a strided (possibly non-contiguous) MxN grid of some fixed struct
// type, summing one double-valued field per row -- the pybind11
// buffer-protocol pattern: a NumPy slice or transpose gives you a raw
// pointer plus explicit byte strides, and you must NOT assume the data is
// packed contiguously.
//
//   buf          -- raw byte buffer holding the grid. It may be (and in
//                    some scenarios here IS) a strided VIEW into a larger
//                    allocation -- do not assume strideRow == N * strideCol.
//   M, N         -- grid shape.
//   strideRow    -- byte distance between element (i, j) and (i+1, j).
//   strideCol    -- byte distance between element (i, j) and (i, j+1).
//   fieldOffset  -- byte offset of the target field WITHIN one struct
//                    element (e.g. offsetof(Elem, val)). May be nonzero.
//   out          -- caller-owned array of length M; out[i] must receive
//                    sum_j value(i, j) where value(i, j) is the `double` at
//                    buf + i*strideRow + j*strideCol + fieldOffset (read via
//                    memcpy -- the address is not guaranteed aligned for a
//                    direct dereference).
//
// out[i] = sum over j in [0, N) of the double at
//          buf + i*strideRow + j*strideCol + fieldOffset
void stridedRowSums(const uint8_t* buf, int M, int N,
                     long strideRow, long strideCol, long fieldOffset,
                     double* out);
