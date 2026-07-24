import heapq

def k_smallest_indices(arr: list[float], k: int) -> list[int]:
    if k == 0:
        return []
    # Use a heap of size k to store (-val, index)
    # Since heapq is a min-heap, we negate the values to simulate a max-heap
    heap = []
    for i, val in enumerate(arr):
        if len(heap) < k:
            heapq.heappush(heap, (-val, i))
        else:
            if -val > heap[0][0]:
                heapq.heapreplace(heap, (-val, i))
    
    return [item[1] for item in heap]
