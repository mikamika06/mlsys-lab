class Block:
    def __init__(self, addr, size, is_allocated, segment_id):
        self.addr = addr
        self.size = size
        self.is_allocated = is_allocated
        self.segment_id = segment_id


class CachingAllocator:
    def __init__(self, segment_size=2097152):
        self.default_segment_size = segment_size
        self.segments = []
        self.blocks = []
        self.next_addr = 0
        self.handles = {}
        self.next_handle = 1
        self.current_allocated = 0
        self.current_reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.peak_fragmentation = 0
        self.reserved_deltas = []

    def malloc(self, size: int) -> int:
        if size <= 0:
            raise ValueError("Allocation size must be positive")

        best_block_idx = None
        best_size = float("inf")

        for i, block in enumerate(self.blocks):
            if not block.is_allocated and block.size >= size:
                if block.size < best_size:
                    best_size = block.size
                    best_block_idx = i

        if best_block_idx is not None:
            block = self.blocks[best_block_idx]
            if block.size == size:
                block.is_allocated = True
                handle = self.next_handle
                self.next_handle += 1
                self.handles[handle] = block
            else:
                rem_size = block.size - size
                block.size = size
                block.is_allocated = True
                handle = self.next_handle
                self.next_handle += 1
                self.handles[handle] = block

                rem_block = Block(
                    block.addr + size, rem_size, False, block.segment_id
                )
                self.blocks.insert(best_block_idx + 1, rem_block)
        else:
            seg_size = max(self.default_segment_size, size)
            seg_id = len(self.segments)
            self.segments.append((self.next_addr, seg_size))
            self.reserved_deltas.append(seg_size)
            self.current_reserved += seg_size

            if seg_size == size:
                block = Block(self.next_addr, size, True, seg_id)
                self.blocks.append(block)
                self.next_addr += seg_size
                handle = self.next_handle
                self.next_handle += 1
                self.handles[handle] = block
            else:
                block = Block(self.next_addr, size, True, seg_id)
                rem_block = Block(
                    self.next_addr + size, seg_size - size, False, seg_id
                )
                self.blocks.extend([block, rem_block])
                self.next_addr += seg_size
                handle = self.next_handle
                self.next_handle += 1
                self.handles[handle] = block

        self.current_allocated += size
        if self.current_allocated > self.peak_allocated:
            self.peak_allocated = self.current_allocated
        if self.current_reserved > self.peak_reserved:
            self.peak_reserved = self.current_reserved

        frag = self.current_reserved - self.current_allocated
        if frag > self.peak_fragmentation:
            self.peak_fragmentation = frag

        return handle

    def free(self, handle: int) -> None:
        if handle not in self.handles:
            raise KeyError("Invalid handle")
        block = self.handles[handle]
        if not block.is_allocated:
            raise ValueError("Block already freed")

        block.is_allocated = False
        self.current_allocated -= block.size
        del self.handles[handle]

        frag = self.current_reserved - self.current_allocated
        if frag > self.peak_fragmentation:
            self.peak_fragmentation = frag

        self.coalesce()

    def coalesce(self) -> None:
        i = 0
        while i < len(self.blocks) - 1:
            b1 = self.blocks[i]
            b2 = self.blocks[i + 1]
            if (
                not b1.is_allocated
                and not b2.is_allocated
                and b1.segment_id == b2.segment_id
                and b1.addr + b1.size == b2.addr
            ):
                b1.size += b2.size
                self.blocks.pop(i + 1)
            else:
                i += 1
