#include "sol.hpp"

TlbVerdict classify_tlb_thrash(int extent, long long stride_bytes, int elem_bytes,
                                int page_bytes, int tlb_entries) {
    // your code here
    (void)extent;
    (void)stride_bytes;
    (void)elem_bytes;
    (void)page_bytes;
    (void)tlb_entries;
    return TlbVerdict::NoThrash;
}
