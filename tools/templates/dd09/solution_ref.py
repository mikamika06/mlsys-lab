def compact_insert(keys, size):
    mask = size - 1
    entries = []
    indices = [-1] * size
    for key in keys:
        perturb = key
        i = key & mask
        while indices[i] != -1 and entries[indices[i]] != key:
            i = (i * 5 + perturb + 1) & mask
            perturb >>= 5
        if indices[i] == -1:
            entries.append(key)
            indices[i] = len(entries) - 1
    return entries, indices
