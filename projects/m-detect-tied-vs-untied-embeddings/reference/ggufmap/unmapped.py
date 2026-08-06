def find_unmapped(hf_keys, mapped_keys):
    """Find unmapped keys."""
    mapped_set = set(mapped_keys)
    return sorted([k for k in hf_keys if k not in mapped_set])
