from mlsys.sim import cache as cachesim

def tlb_miss_count(addrs, page_size):
    """Return the number of TLB misses for the given byte addresses and page size.
    
    TLB: 64 sets, 4 ways, LRU.
    """
    raise NotImplementedError('your code here')
