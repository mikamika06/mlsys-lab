#pragma once

// Given the byte sizes of n struct fields (each field's natural alignment
// equals its size -- true for char/short/int/long/float/double/pointer
// under the LP64 ABI), compute the SMALLEST possible sizeof() achievable
// by freely reordering those exact fields (any permutation) into a single
// struct, under the standard C++ layout rule: each field is placed at the
// next offset that is a multiple of its own size, and the struct's total
// size is then rounded up to a multiple of the largest field size present.
int minimal_sizeof(const int* sizes, int n);
