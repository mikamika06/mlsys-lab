class GGUFIterator:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "rb")
        self.offset = 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.file.read(64)
        if not chunk:
            self.file.close()
            raise StopIteration
        return chunk
