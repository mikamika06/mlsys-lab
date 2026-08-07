import heapq

class OverloadQueue:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.heap = []
        self.counter = 0

    def push(self, item, priority: int = 0) -> bool:
        if len(self.heap) >= self.capacity:
            return False
        heapq.heappush(self.heap, (-priority, self.counter, item))
        self.counter += 1
        return True

    def pop(self):
        if not self.heap:
            return None
        _, _, item = heapq.heappop(self.heap)
        return item

    def size(self) -> int:
        return len(self.heap)
