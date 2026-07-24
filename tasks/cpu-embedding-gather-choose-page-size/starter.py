from mlsys.sim import cache as cachesim

def choose_page_size(indices, row_bytes, page_sizes):
    """Return the page size that produces the fewest simulated TLB misses."""
    raise NotImplementedError("implement me")
