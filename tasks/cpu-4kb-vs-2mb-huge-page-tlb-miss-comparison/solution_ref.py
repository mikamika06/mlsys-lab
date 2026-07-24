from mlsys.sim import cache as cachesim

def tlb_miss_count(addrs, page_size):
    """Return number of TLB misses for given addresses and page size."""
    result = cachesim.simulate(addrs, line_bytes=page_size, sets=64, ways=4)
    return result["misses"]
