#pragma once

// ---------------------------------------------------------------------------
// Implement the SAME "does x+1 overflow a 32-bit signed int?" check twice,
// so the two entry points differ ONLY in how they get compiled:
//
//   check_no_overflow_noopt : marked __attribute__((optnone)) -- the
//                              compiler is forced to emit UNOPTIMIZED code
//                              for this function specifically, regardless
//                              of the file's real optimization level. This
//                              is "what -O0 does" for this exact logic.
//
//   check_no_overflow_opt   : compiled normally, at this file's real -O2
//                              level. This is "what -O2 does".
//
// Both must return false when x + 1 would signed-overflow (i.e. x ==
// INT_MAX), and true otherwise. Critically, BOTH must agree with each
// other on EVERY input -- the check must survive optimization, not just
// happen to work at one optimization level.
// ---------------------------------------------------------------------------
__attribute__((optnone)) bool check_no_overflow_noopt(int x);
bool check_no_overflow_opt(int x);
