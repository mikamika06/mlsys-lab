import hashlib


def _hash_chunk(chunk):
    h = hashlib.sha256()
    for token in chunk:
        h.update(int(token).to_bytes(8, "little", signed=True))
    return h.digest()


def cacheblend_plan(cached_docs, new_doc, chunk_size):
    cache = {}
    for doc_index, doc in enumerate(cached_docs):
        for start in range(0, len(doc) - chunk_size + 1, chunk_size):
            chunk = tuple(doc[start:start + chunk_size])
            key = _hash_chunk(chunk)
            if key not in cache:
                cache[key] = (doc_index, start // chunk_size)

    reuse = []
    recompute = set()
    for new_chunk_index, start in enumerate(range(0, len(new_doc) - chunk_size + 1, chunk_size)):
        chunk = tuple(new_doc[start:start + chunk_size])
        key = _hash_chunk(chunk)
        if key in cache:
            old_doc, old_chunk = cache[key]
            reuse.append([new_chunk_index, old_doc, old_chunk])
            recompute.add(start)
            end = start + chunk_size - 1
            if end < len(new_doc):
                recompute.add(end)

    return {
        "reuse": reuse,
        "recompute": sorted(recompute),
    }
