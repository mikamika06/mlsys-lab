from typing import Callable, Any, Optional, Dict, Tuple


class DummyBlockMask:
    """Simulated BlockMask tensor object holding structural block grid metadata."""

    def __init__(
        self,
        shape: Tuple[int, int],
        block_size: int,
        active_blocks: int,
        mask_fn_id: str,
    ):
        self.shape = shape
        self.block_size = block_size
        self.active_blocks = active_blocks
        self.mask_fn_id = mask_fn_id


class MaskCache:
    """Amortized mask caching policy for FlexAttention BlockMask instances."""

    def __init__(self, max_capacity: int = 16):
        self.max_capacity = max_capacity

    def get_or_create(
        self,
        shape: Tuple[int, int],
        block_size: int,
        mask_fn_id: str,
        builder_fn: Callable[[], DummyBlockMask],
    ) -> Tuple[DummyBlockMask, bool]:
        """Retrieve cached BlockMask or construct a new one.

        Returns (block_mask, is_cache_hit).
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cached entries."""
        raise NotImplementedError

    def size(self) -> int:
        """Return current number of entries in cache."""
        raise NotImplementedError
