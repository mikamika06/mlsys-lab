def build_radix_tree(sequences: list) -> list:
    """Insert `sequences` in order into a compressed-prefix (radix) tree,
    splitting an edge at the divergence point whenever a new insert shares
    only part of an existing edge. Return a sorted list of
    (parent_path, edge_tokens) tuples (both tuples of ints) describing
    every edge in the resulting tree."""
    raise NotImplementedError('your code here')
