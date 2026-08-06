def load_plan(blob, page=16384, want=None):
    """What faulting these tensors in through mmap would actually cost.

    Returns page_size, file_bytes, metadata_bytes, weight_bytes,
    metadata_fraction, selected_tensors, distinct_pages, resident_bytes,
    resident_fraction, shared_pages, and a per-tensor list of
    {name, byte_range, page_range, pages, resident_bytes, waste_bytes}.

    want=None means every tensor. Two tensors that share a page must be counted
    once in distinct_pages and once in resident_bytes.
    """
    raise NotImplementedError


def alignment_report(blob, page=16384):
    """Per tensor: its offset, whether it is aligned to the container's
    alignment, whether it is page-aligned, and how far into a page it starts."""
    raise NotImplementedError
