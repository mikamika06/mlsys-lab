import hashlib


def _chunk_hash(chunk):
    h = hashlib.sha256()
    for token in chunk:
        h.update(int(token).to_bytes(8, "little", signed=True))
    return h.digest()


def _oracle(cached_docs, new_doc, chunk_size):
    cache = {}
    for doc_index, doc in enumerate(cached_docs):
        for chunk_index in range(0, len(doc) - chunk_size + 1, chunk_size):
            chunk = tuple(doc[chunk_index:chunk_index + chunk_size])
            key = _chunk_hash(chunk)
            if key not in cache:
                cache[key] = (doc_index, chunk_index // chunk_size)

    reuse = []
    reused_chunks = []
    for new_chunk_index, start in enumerate(range(0, len(new_doc) - chunk_size + 1, chunk_size)):
        chunk = tuple(new_doc[start:start + chunk_size])
        key = _chunk_hash(chunk)
        if key in cache:
            old_doc, old_chunk = cache[key]
            reuse.append([new_chunk_index, old_doc, old_chunk])
            reused_chunks.append(new_chunk_index)

    recompute = set()
    for chunk_index in reused_chunks:
        start = chunk_index * chunk_size
        recompute.add(start)
        end = start + chunk_size - 1
        if end < len(new_doc):
            recompute.add(end)

    return {
        "reuse": reuse,
        "recompute": sorted(recompute),
    }


def grade(sol, fx) -> dict:
    cases = [
        (
            [[1, 2, 3, 4, 9, 9], [8, 7, 6, 5]],
            [8, 7, 6, 5, 1, 2, 3, 4],
            4,
        ),
        (
            [[10, 11, 12, 13, 20, 21, 22, 23], [1, 2, 3, 4]],
            [20, 21, 22, 23, 9, 9, 1, 2, 3, 4, 10, 11, 12, 13],
            4,
        ),
        (
            [[5, 6, 7, 8], [5, 6, 7, 8], [1, 1, 1, 1]],
            [5, 6, 7, 8, 5, 6, 7, 8],
            4,
        ),
        (
            [[3, 4, 5, 6, 7, 8], [9, 10, 11, 12]],
            [7, 8, 9, 10, 11, 12, 3, 4, 5, 6],
            2,
        ),
    ]

    ok = 1.0
    for cached_docs, new_doc, chunk_size in cases:
        try:
            got = sol.cacheblend_plan(cached_docs, new_doc, chunk_size)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(cached_docs, new_doc, chunk_size):
            ok = 0.0
            break
    return {"exact_match": ok}
