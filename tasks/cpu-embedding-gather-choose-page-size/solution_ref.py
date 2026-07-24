from mlsys.sim import cache as cachesim

def choose_page_size(indices, row_bytes, page_sizes):
    addrs = [i * row_bytes for i in indices]
    best = None
    best_misses = None
    for page_bytes in page_sizes:
        res = cachesim.simulate(addrs, line_bytes=page_bytes, sets=64, ways=1)
        misses = res["misses"]
        if best is None or misses < best_misses or (misses == best_misses and page_bytes < best):
            best = page_bytes
            best_misses = misses
    return best
