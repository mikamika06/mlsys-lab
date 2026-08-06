def estimate_resident_bytes(file_size, accesses, page_size=4096):
    """Estimate page-cache resident size for given byte access ranges."""
    resident_pages = set()
    for offset, length in accesses:
        if length <= 0 or offset >= file_size:
            continue
        end_offset = min(offset + length, file_size)
        start_page = offset // page_size
        end_page = (end_offset - 1) // page_size
        for page_idx in range(start_page, end_page + 1):
            resident_pages.add(page_idx)
    return len(resident_pages) * page_size
