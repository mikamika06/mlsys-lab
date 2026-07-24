from mlsys.sim import cache as cachesim

def _simulate_tlb(addrs, page_size):
    """Return number of TLB misses for given addresses and page size."""
    # The TLB simulator expects byte addresses; page-aligned within simulator
    result = cachesim.simulate(addrs, line_bytes=page_size, sets=64, ways=4)
    return result["misses"]

def grade(sol, fx) -> dict:
    # Deterministic address trace: 8 KB working set spread over 2 MB range
    test_addrs = [0, 4096, 8192, 12288, 16384, 20480, 24576, 28672,
                  32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440,
                  1048576, 1048576, 1048576, 1048576]  # 16 unique 4K pages + repeated huge-page region
    
    # Reference: compute with the simulator itself
    miss_4k_ref = _simulate_tlb(test_addrs, 4096)
    miss_huge_ref = _simulate_tlb(test_addrs, 2 * 1024 * 1024)
    
    try:
        miss_4k, miss_huge = sol.tlb_miss_count(test_addrs, 4096), sol.tlb_miss_count(test_addrs, 2 * 1024 * 1024)
    except Exception:
        return {"exact_match": 0.0}
    
    # Forces integer comparison
    ok = (miss_4k == miss_4k_ref) and (miss_huge == miss_huge_ref)
    return {"exact_match": 1.0 if ok else 0.0}
