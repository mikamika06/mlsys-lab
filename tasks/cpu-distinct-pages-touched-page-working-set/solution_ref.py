import numpy as np

def count_distinct_pages(trace: np.ndarray, page_size: int = 4096) -> int:
    # Ensure trace is an array of integers
    arr = np.asarray(trace, dtype=np.uint64)
    pages = arr // page_size
    return int(np.unique(pages).size)
