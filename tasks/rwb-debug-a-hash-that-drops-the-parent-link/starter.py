def block_salted_hash(stream, salt):
    """Return a list of parent-linked salted hashes for each block in the stream.

    BUG: The parent link is dropped — only block and salt are hashed.
    """
    result = []
    for block in stream:
        h = hash((block, salt))  # BUG: missing parent link h
        result.append(h)
    return result
