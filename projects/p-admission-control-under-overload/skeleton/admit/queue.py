class OverloadQueue:
    def __init__(self, capacity: int):
        raise NotImplementedError

    def push(self, item, priority: int = 0) -> bool:
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError
