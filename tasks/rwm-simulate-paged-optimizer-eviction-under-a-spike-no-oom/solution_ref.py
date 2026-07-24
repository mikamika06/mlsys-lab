from collections import OrderedDict


def simulate_paged_eviction(trace, budget_pages):
    resident = OrderedDict()
    fault_count = 0
    evicted = []
    for page in trace:
        if page in resident:
            resident.move_to_end(page)
        else:
            fault_count += 1
            if len(resident) >= budget_pages:
                evict_page, _ = resident.popitem(last=False)
                evicted.append(evict_page)
            resident[page] = True
    return {
        "fault_count": fault_count,
        "evicted_pages": evicted,
        "final_resident": list(resident.keys()),
    }
