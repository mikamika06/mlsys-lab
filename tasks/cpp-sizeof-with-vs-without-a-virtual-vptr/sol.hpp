#pragma once

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Under the Itanium C++ ABI, giving a class at least one virtual function
// is equivalent, for layout purposes, to prepending a hidden vptr field
// (8 bytes, 8-byte alignment) before all the user-declared fields.
//
// Given the REAL, compiler-computed `sizeof` (`plain_size`) and `alignof`
// (`plain_align`) of a plain (non-virtual) aggregate, compute the
// `sizeof` the SAME fields would have if the class also had a virtual
// function:
//
//   1. The vptr occupies the first 8 bytes; every user field's offset
//      shifts forward by exactly 8 (valid because 8 is a multiple of
//      every individual field alignment in this ABI's primitive set
//      {1,2,4,8}, so all the plain struct's internal padding is
//      unaffected by the shift).
//   2. The struct's overall alignment becomes max(plain_align, 8).
//   3. The final size is padded up to a multiple of that new alignment.
//
// Return the resulting sizeof for the polymorphic version.
// ---------------------------------------------------------------------------
long virtual_sizeof(long plain_size, long plain_align);
