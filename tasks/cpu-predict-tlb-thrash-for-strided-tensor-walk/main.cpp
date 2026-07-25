#include <cstdio>
#include "sol.hpp"

// FIXED driver. 4 hand-picked walks (no rand()/time()), all against a
// 16 KiB page size (Apple Silicon's real page size) and a 128-entry TLB.

namespace {
void run(const char* name, int extent, long long stride_bytes, int elem_bytes,
         int page_bytes, int tlb_entries) {
    TlbVerdict v = classify_tlb_thrash(extent, stride_bytes, elem_bytes, page_bytes, tlb_entries);
    printf("%s extent=%d stride=%lld verdict=%d\n", name, extent, stride_bytes,
           static_cast<int>(v));
}
}  // namespace

int main() {
    constexpr int kPage = 16384;  // 16 KiB
    constexpr int kTlb = 128;     // 128-entry fully-associative TLB

    // A: contiguous float32 sweep, modest size (~390 KB span).
    run("contiguous_small", 100000, 4, 4, kPage, kTlb);

    // B: contiguous float32 sweep, huge size (~38 MB span).
    run("contiguous_huge", 10000000, 4, 4, kPage, kTlb);

    // C: column walk of a 4096-column row-major float32 matrix -- stride
    // equals exactly one page, so every single row lands on a new page.
    run("column_walk_page_stride", 1000, 16384, 4, kPage, kTlb);

    // D: a dilated/strided tensor view whose step is 2 full pages (e.g.
    // every other row of a very wide matrix) -- every element lands on a
    // page, but the SKIPPED pages in between are never touched at all, so
    // the working set is far smaller than the address span alone suggests.
    run("large_stride_skips_pages", 100, 32768, 4, kPage, kTlb);

    // E: modest strided walk (e.g. a narrow 16-column matrix's column) --
    // small enough working set to stay resident.
    run("small_stride_fits", 500, 64, 4, kPage, kTlb);

    return 0;
}
