def classify_heap_types(types):
    heap_mask = 1 << 9
    return [bool(t.__flags__ & heap_mask) for t in types]
