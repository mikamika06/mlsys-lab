import json

class Simulator:
    def __init__(self, max_split_size=float('inf')):
        self.max_split_size = max_split_size
        self.segments = []
        self.next_id = 1

    def allocate(self, size):
        best_seg, best_i, best_b = None, -1, None
        for seg in self.segments:
            for i, b in enumerate(seg["blocks"]):
                if b["state"] == "free" and b["size"] >= size:
                    if self.max_split_size and b["size"] > self.max_split_size and size <= self.max_split_size:
                        continue
                    if best_b is None or b["size"] < best_b["size"]:
                        best_seg, best_i, best_b = seg, i, b

        if best_b is None:
            seg_size = max(size, 20 if size <= 20 else 100)
            new_seg = {"size": seg_size, "blocks": [{"id": 0, "size": seg_size, "state": "free"}]}
            self.segments.append(new_seg)
            best_seg = new_seg
            best_i = 0
            best_b = new_seg["blocks"][0]

        b = best_b
        if b["size"] == size:
            b["state"] = "allocated"
            b["id"] = self.next_id
            self.next_id += 1
            return b["id"]
        else:
            new_b = {"id": self.next_id, "size": size, "state": "allocated"}
            self.next_id += 1
            rem = {"id": 0, "size": b["size"] - size, "state": "free"}
            best_seg["blocks"].pop(best_i)
            best_seg["blocks"].insert(best_i, rem)
            best_seg["blocks"].insert(best_i, new_b)
            return new_b["id"]

    def free(self, id):
        for seg in self.segments:
            for i, b in enumerate(seg["blocks"]):
                if b["id"] == id and b["state"] == "allocated":
                    b["state"] = "free"
                    b["id"] = 0
                    self._merge(seg)
                    return

    def _merge(self, seg):
        i = 0
        while i < len(seg["blocks"]) - 1:
            if seg["blocks"][i]["state"] == "free" and seg["blocks"][i+1]["state"] == "free":
                seg["blocks"][i]["size"] += seg["blocks"][i+1]["size"]
                seg["blocks"].pop(i+1)
            else:
                i += 1

    def get_snapshot(self):
        import copy
        return {"segments": copy.deepcopy(self.segments)}

    def reserved_memory(self):
        return sum(seg["size"] for seg in self.segments)

def create_mock_snapshot(path):
    data = {
        "segments": [
            {
                "size": 100,
                "blocks": [
                    {"id": 1, "size": 40, "state": "allocated"},
                    {"id": 0, "size": 60, "state": "free"}
                ]
            }
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)

def generate_test_snapshots():
    s = Simulator()
    id1 = s.allocate(30)
    id2 = s.allocate(40)
    snap1 = s.get_snapshot()
    s.free(id1)
    id3 = s.allocate(10)
    snap2 = s.get_snapshot()
    return [snap1, snap2]

def run_workload(max_split_size, steps=1000):
    import random
    rng = random.Random(42)
    sim = Simulator(max_split_size)
    retained = []
    peak_res = 0
    for _ in range(steps):
        l_size = rng.choice([30, 40, 50])
        l1 = sim.allocate(l_size)
        s1 = sim.allocate(10)
        l2_size = rng.choice([30, 40, 50])
        l2 = sim.allocate(l2_size)
        s2 = sim.allocate(10)

        sim.free(l1)
        sim.free(l2)
        retained.append(s1)
        retained.append(s2)
        if len(retained) > 20:
            sim.free(retained.pop(0))
            sim.free(retained.pop(0))
        peak_res = max(peak_res, sim.reserved_memory())
    return peak_res
