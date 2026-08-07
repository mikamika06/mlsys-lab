class MemorySnapshotAnalyzer:
    def __init__(self, snapshot_data):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError

    def compute_fragmentation(self):
        raise NotImplementedError

    def find_leaks(self):
        raise NotImplementedError
