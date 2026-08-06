import numpy as np

def count_distinct_pages(trace: np.ndarray, page_size: int = 4096) -> int:
    arr = np.asarray(trace, dtype=np.uint64)
    seen = []
    for i in range(len(arr)):
        page = int(arr[i]) // page_size
        found = False
        for j in range(len(seen)):
            if seen[j] == page:
                found = True
                break
        if not found:
            seen.append(page)
    return len(seen)
