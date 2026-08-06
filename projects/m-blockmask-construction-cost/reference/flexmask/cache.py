from typing import Callable, Tuple, Dict, Optional, List


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
        self._cache: Dict[Tuple[Tuple[int, int], int, str], DummyBlockMask] = {}
        self._usage_order: List[Tuple[Tuple[int, int], int, str]] = []

    def get_or_create(
        self,
        shape: Tuple[int, int],
        block_size: int,
        mask_fn_id: str,
        builder_fn: Callable[[], DummyBlockMask],
    ) -> Tuple[DummyBlockMask, bool]:
        """Retrieve cached BlockMask or construct a new one."""
        key = (shape, block_size, mask_fn_id)

        if key in self._cache:
            self._usage_order.remove(key)
            self._usage_order.append(key)
            return self._cache[key], True

        mask = builder_fn()
        if len(self._cache) >= self.max_capacity:
            lru_key = self._usage_order.pop(0)
            del self._cache[lru_key]

        self._cache[key] = mask
        self._usage_order.append(key)
        return mask, False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._usage_order.clear()

    def size(self) -> int:
        """Return current number of entries in cache."""
        return len(self._cache)
