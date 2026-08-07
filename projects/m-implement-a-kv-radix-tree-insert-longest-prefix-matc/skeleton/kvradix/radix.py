class RadixNode:
    """Radix tree node representing a sequence of token IDs."""

    def __init__(self, key=None):
        """Initialize a RadixNode."""
        raise NotImplementedError


class RadixTree:
    """Radix tree for managing KV cache prefix matching."""

    def __init__(self):
        """Initialize an empty RadixTree."""
        raise NotImplementedError

    def match_prefix(self, tokens):
        """Find the longest prefix match for a token sequence.

        Returns (matched_length, matched_node, remaining_tokens).
        """
        raise NotImplementedError

    def insert(self, tokens, value=None):
        """Insert a token sequence into the radix tree.

        Returns the leaf node representing the end of the inserted sequence.
        """
        raise NotImplementedError
