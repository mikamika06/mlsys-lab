import hashlib

def block_hashes(tokens, block_size):
    """
    Given a list of integer tokens and a block size, return a list of hex strings
    representing the chained SHA-256 hash of each full block.
    The hash of block N must include the hash digest of block N-1.
    """
    raise NotImplementedError

def reusable_blocks(hashes1, hashes2):
    """
    Given two lists of block hashes, return the number of reusable blocks
    (the length of the matching prefix of hashes).
    """
    raise NotImplementedError

def divergence(tokens1, tokens2, block_size):
    """
    Return a tuple (divergence_token_index, lost_block_count).
    divergence_token_index is the index of the first token that differs.
    lost_block_count is the number of full blocks in tokens2 that have the exact
    same tokens as the corresponding block in tokens1, but occur at or after the
    divergence token (so their chained hash differs and they cannot be reused).
    """
    raise NotImplementedError
