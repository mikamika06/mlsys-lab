#pragma once

// ============================================================================
// Deterministic set-mapping oracle (FIXED — defined in main.cpp). A cache
// with `line_bytes`-byte lines and `sets` sets maps byte address `addr` to
// set (addr / line_bytes) % sets. This is the ground truth the driver uses
// to VERIFY your derived stride -- it does not depend on your answer.
// ============================================================================
int set_of(long addr, int line_bytes, int sets);

// ============================================================================
// For a cache with `line_bytes`-byte lines and `sets` sets (any
// associativity -- the number of ways affects only how many colliding
// lines survive before eviction, never which set an address maps to, so it
// is not a parameter here), derive the MINIMUM positive stride S such that
// every address in the sequence 0, S, 2S, 3S, ... maps to the SAME set as
// address 0.
// ============================================================================
long collision_stride(int line_bytes, int sets);
