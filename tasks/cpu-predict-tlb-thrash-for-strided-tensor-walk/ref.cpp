#include <unordered_set>
#include "sol.hpp"

TlbVerdict classify_tlb_thrash(int extent, long long stride_bytes, int elem_bytes,
                                int page_bytes, int tlb_entries) {
    (void)elem_bytes;
    std::unordered_set<long long> pages;
    pages.reserve(static_cast<size_t>(extent));
    for (int i = 0; i < extent; ++i) {
        long long addr = static_cast<long long>(i) * stride_bytes;
        pages.insert(addr / page_bytes);
    }
    return (static_cast<int>(pages.size()) > tlb_entries) ? TlbVerdict::Thrash
                                                            : TlbVerdict::NoThrash;
}
