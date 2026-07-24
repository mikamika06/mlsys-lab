BASE = 1_000_003
MOD = (1 << 61) - 1


def _block_hash(parent: int, tokens) -> int:
    """Chained rolling hash of one block: folds the parent block's hash
    together with every token in this block, so a block's hash depends
    on its ENTIRE preceding chain, not just its own content."""
    h = parent
    for t in tokens:
        h = (h * BASE + t + 1) % MOD
    return h


def longest_cached_prefix_blocks(cached_hashes: set, new_tokens: list, block_size: int) -> int:
    """Prefix-cache lookup (vLLM APC / SGLang RadixAttention style, block
    granularity): compute the chained hash of each of `new_tokens`'s FULL
    blocks in order, using block 0's parent hash = 0, and each later
    block's parent = the previous block's chain hash. Walk the blocks in
    order and count how many CONSECUTIVE leading blocks have their chain
    hash present in `cached_hashes`, stopping at the first miss.

    A trailing block with fewer than `block_size` tokens is never a hit
    (real block caches only ever store complete blocks) and, being past
    the end anyway, also stops the walk.

    cached_hashes : set (or any container supporting `in`) of int block
                    chain hashes currently resident in the cache.
    new_tokens    : list of int token ids for the new request.
    block_size    : positive int.

    Returns the number of leading blocks that hit (an int in
    `[0, len(new_tokens) // block_size]`).
    """
    raise NotImplementedError('your code here')
