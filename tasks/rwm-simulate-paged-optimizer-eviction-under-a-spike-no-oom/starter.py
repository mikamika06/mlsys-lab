def simulate_paged_eviction(trace: list[int], budget_pages: int) -> dict:
    """Simulate CUDA-unified-memory-style paging of optimizer-state
    pages between GPU (resident) and CPU, under an LRU eviction policy,
    so a memory spike never fails an allocation.

    trace: list of page ids (ints) accessed in order -- e.g. each Adam
        step touches a handful of optimizer-state pages, and a spike
        (batch-size jump, activation checkpoint burst, etc.) can touch
        far more distinct pages than fit in the GPU budget at once.
    budget_pages: max number of pages allowed resident on GPU at the
        same time.

    Policy: on each access,
      - if the page is already resident, it becomes the most-recently-used
        page (a "hit", no fault).
      - otherwise it's a "fault": if the resident set is already at
        `budget_pages`, evict the least-recently-used resident page
        first (moving it to CPU), then bring the requested page in as
        the most-recently-used page. Because eviction always happens
        before the new page is added, the resident set never exceeds
        `budget_pages` -- no allocation ever fails, even mid-spike.

    Returns a dict:
      {
        "fault_count": int,            # total faults over the whole trace
        "evicted_pages": list[int],    # page ids evicted, IN EVICTION ORDER
        "final_resident": list[int],   # pages resident at the end, LRU..MRU order
      }
    """
    raise NotImplementedError('your code here')
