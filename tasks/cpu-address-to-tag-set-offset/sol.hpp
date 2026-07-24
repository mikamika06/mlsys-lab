#pragma once

struct AddrDecomp {
    unsigned long tag;
    unsigned long set_index;
    unsigned long offset;
};

// Decompose a byte address into (tag, set_index, offset) for a cache with
// `line_bytes` bytes per line and `sets` sets (`ways` is accepted for API
// completeness — it does not affect the decomposition, only how many
// lines can live in one set):
//   offset    = addr % line_bytes
//   set_index = (addr / line_bytes) % sets
//   tag       = (addr / line_bytes) / sets
AddrDecomp decompose_address(unsigned long addr, int line_bytes, int sets, int ways);
