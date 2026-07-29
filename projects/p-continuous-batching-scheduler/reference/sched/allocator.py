class Allocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.ref = [0] * num_blocks
        self._by_key = {}
        self._by_block = {}

    def allocate(self) -> int:
        if not self.free:
            raise RuntimeError("out of blocks")
        b = self.free.pop(0)
        self.ref[b] = 1
        return b

    def share(self, block: int) -> int:
        self.ref[block] += 1
        return block

    def release(self, block: int) -> None:
        if self.ref[block] == 0:
            return
        self.ref[block] -= 1
        if self.ref[block] == 0:
            key = self._by_block.pop(block, None)
            if key is not None and self._by_key.get(key) == block:
                del self._by_key[key]
            self.free.append(block)
            self.free.sort()

    def free_count(self) -> int:
        return len(self.free)

    def register(self, block: int, key: str) -> None:
        self._by_key[key] = block
        self._by_block[block] = key

    def lookup(self, key: str):
        return self._by_key.get(key)
