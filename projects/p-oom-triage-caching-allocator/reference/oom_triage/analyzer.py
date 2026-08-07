import json

def load_snapshot(path):
    with open(path, "r") as f:
        return json.load(f)

def analyze_fragmentation(snapshot):
    allocated = 0
    reserved = 0
    max_free = 0
    for seg in snapshot.get("segments", []):
        reserved += seg["size"]
        for block in seg["blocks"]:
            if block["state"] == "allocated":
                allocated += block["size"]
            elif block["state"] == "free":
                max_free = max(max_free, block["size"])
    return {"allocated": allocated, "reserved": reserved, "max_free": max_free}

def find_leaked_tensors(snapshots):
    if not snapshots:
        return []
    common = None
    for snap in snapshots:
        current = set()
        for seg in snap.get("segments", []):
            for block in seg["blocks"]:
                if block["state"] == "allocated":
                    current.add(block["id"])
        if common is None:
            common = current
        else:
            common.intersection_update(current)
    return sorted(list(common))
