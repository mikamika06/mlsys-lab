def fragmentation_waste_fraction(lengths, page_size):
    total = sum(lengths)
    paged_total = 0
    for length in lengths:
        pages = (length + page_size - 1) // page_size
        paged_total += pages * page_size

    pre_total = len(lengths) * max(lengths)

    paged_fraction = (paged_total - total) / paged_total
    preallocated_fraction = (pre_total - total) / pre_total

    return paged_fraction, preallocated_fraction
