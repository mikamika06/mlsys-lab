class MemorySnapshotAnalyzer:
    def __init__(self, snapshot_data):
        self.data = snapshot_data

    def parse(self):
        if not isinstance(self.data, dict) or "segments" not in self.data:
            raise ValueError("Invalid snapshot format")
        return len(self.data["segments"]) > 0

    def compute_fragmentation(self):
        total_reserved = sum(s.get("size", 0) for s in self.data.get("segments", []))
        total_allocated = sum(b.get("size", 0) for s in self.data.get("segments", []) for b in s.get("blocks", []))
        if total_reserved == 0:
            return 0.0
        frag = 1.0 - (total_allocated / total_reserved)
        return frag

    def find_leaks(self):
        leaks = []
        for s in self.data.get("segments", []):
            for b in s.get("blocks", []):
                if b.get("state") == "active" and b.get("persistent", False):
                    leaks.append(b.get("id"))
        return leaks
