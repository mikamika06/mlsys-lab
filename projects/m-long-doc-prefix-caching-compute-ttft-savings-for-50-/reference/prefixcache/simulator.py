import hashlib


def compute_block_hashes(tokens: list[int], block_size: int) -> list[int]:
    """Compute deterministic chained prefix hashes for block-aligned sub-sequences."""
    num_blocks = len(tokens) // block_size
    hashes = []
    prev = 0
    for i in range(num_blocks):
        block = tokens[i * block_size : (i + 1) * block_size]
        data = f"{prev}:" + ",".join(map(str, block))
        h = int(hashlib.md5(data.encode("utf-8")).hexdigest()[:8], 16)
        hashes.append(h)
        prev = h
    return hashes


def simulate_prefix_cache(
    doc_tokens: list[int],
    question_tokens_list: list[list[int]],
    block_size: int,
) -> list[dict]:
    """Simulate sequential requests over a document prefix with block prefix caching."""
    cache_table = set()
    results = []

    for i, q_tokens in enumerate(question_tokens_list):
        prompt = doc_tokens + q_tokens
        hashes = compute_block_hashes(prompt, block_size)

        matched_blocks = 0
        for h in hashes:
            if h in cache_table:
                matched_blocks += 1
            else:
                break

        cached_tokens = matched_blocks * block_size
        uncached_tokens = len(prompt) - cached_tokens

        for h in hashes:
            cache_table.add(h)

        results.append(
            {
                "request_id": i,
                "total_tokens": len(prompt),
                "cached_tokens": cached_tokens,
                "uncached_tokens": uncached_tokens,
                "hit_ratio": cached_tokens / len(prompt) if len(prompt) > 0 else 0.0,
            }
        )

    return results
