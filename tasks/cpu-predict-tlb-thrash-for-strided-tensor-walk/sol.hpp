#pragma once

// Thrash = the walk touches more distinct pages than the TLB has entries
// for; NoThrash = the whole working set of pages fits.
enum class TlbVerdict : int { NoThrash = 0, Thrash = 1 };

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// classify_tlb_thrash: a one-dimensional tensor walk visits `extent`
// elements, `elem_bytes` bytes each, with consecutive elements
// `stride_bytes` apart (a "strided" walk -- e.g. sweeping one column of a
// row-major matrix, where `stride_bytes` is the matrix's row width in
// bytes, not the element size). The walk starts at address 0 for i=0 and
// visits address `i * stride_bytes` for i in [0, extent).
//
// The system has a fully-associative TLB with `tlb_entries` entries, each
// covering one `page_bytes`-byte page. If the SAME walk is repeated (e.g.
// an outer loop that sweeps this dimension many times), whether the second
// and later sweeps hit or miss the TLB depends on whether every page the
// walk touches can stay resident at once: count the number of DISTINCT
// pages the walk visits, one per element, using only each element's
// FIRST byte (`floor(i * stride_bytes / page_bytes)` for i in [0, extent),
// counting each distinct value once -- `elem_bytes` is carried for realism
// but this simplified model does not treat an element that happens to
// straddle a page boundary as touching two pages). If the distinct-page
// count exceeds `tlb_entries`, later sweeps cannot all hit -> Thrash.
// Otherwise the whole working set of pages fits and stays resident ->
// NoThrash.
// ============================================================================
TlbVerdict classify_tlb_thrash(int extent, long long stride_bytes, int elem_bytes,
                                int page_bytes, int tlb_entries);
