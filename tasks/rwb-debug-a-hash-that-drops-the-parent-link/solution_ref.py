def block_salted_hash(stream, salt):
    """Return a list of parent-linked salted hashes for each block in the stream."""
    h = salt
    result = []
    for block in stream:
        h = hash((block, salt, h))
        result.append(h)
    return result
