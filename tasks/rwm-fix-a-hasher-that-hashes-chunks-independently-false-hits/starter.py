def find_reusable_chunks(trace, chunk_size):
    """Return chunk starts that can reuse previous computation."""
    seen = set()
    reusable = []
    for i in range(len(trace) - chunk_size + 1):
        chunk = tuple(trace[i:i + chunk_size])
        if chunk in seen:
            reusable.append(i)
        else:
            seen.add(chunk)
    return reusable
