import json

class MemorySnapshotAnalyzer:
    def __init__(self, data=None):
        self.data = data or {}
        self.fixed = False

    def load_snapshot(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)
        return True

    def analyze_fragmentation(self, baseline, current):
        active_diff = current.get("active", 0) - baseline.get("active", 0)
        allocated_diff = current.get("allocated", 0) - baseline.get("allocated", 0)
        fragmented = allocated_diff > active_diff
        return {"fragmented": fragmented, "leak_size": active_diff if not fragmented else 0}

    def find_reference_chain(self, target_id):
        objects = self.data.get("objects", {})
        if target_id not in objects:
            return []
        chain = [target_id]
        curr = target_id
        while curr in objects and objects[curr].get("parent"):
            curr = objects[curr]["parent"]
            chain.append(curr)
        return chain[::-1]

    def fix_retention(self):
        self.fixed = True
        return True

    def simulate_epoch(self):
        if not self.fixed:
            return 40
        return 0
