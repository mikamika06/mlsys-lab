#pragma once
#include <cstddef>

// ============================================================================
// A record needs 4 fields: two 8-byte doubles (a, b) and two 1-byte bools
// (flag1, flag2). The BROKEN layout you start from (see solve.cpp) wedges
// each bool directly after a double:
//
//   struct BadRecord {
//       double a;
//       bool   flag1;
//       double b;
//       bool   flag2;
//   };
//
// `a` lands at offset 0, `flag1` at offset 8, then the compiler inserts 7
// bytes of padding before `b` so it lands back on an 8-byte boundary, `b`
// at offset 16, `flag2` at offset 24, then 7 MORE tail bytes to round the
// whole struct up to its own 8-byte alignment -- sizeof(BadRecord) == 32,
// for 18 bytes of actual data.
//
// Fix it: reorder the fields (both doubles together, then both bools
// together) so the compiler only pads once, at the very end, instead of
// after every bool. Report your fixed layout through:
// ============================================================================
size_t record_size();   // sizeof(your fixed record)
size_t offset_a();      // offsetof(your fixed record, a)
size_t offset_b();      // offsetof(your fixed record, b)
size_t offset_flag1();  // offsetof(your fixed record, flag1)
size_t offset_flag2();  // offsetof(your fixed record, flag2)
