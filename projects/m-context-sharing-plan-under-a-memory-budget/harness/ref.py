import random

def generate_fixtures():
    random.seed(42)
    configs = []
    for _ in range(5):
        tensors = [{"id": i, "size": random.randint(100, 2000), "exclusive": random.choice([True, False])} for i in range(10)]
        budget = random.randint(3000, 6000)
        configs.append((tensors, budget))
    return configs

CONFIGS = generate_fixtures()

def build_sharing_plan(tensors, budget):
    sorted_t = sorted(tensors, key=lambda x: x["size"], reverse=True)
    allocated = []
    current_bytes = 0
    for t in sorted_t:
        if current_bytes + t["size"] <= budget:
            allocated.append(t["id"])
            current_bytes += t["size"]
    return sorted(allocated)

def classify_oom(event_log, workspace_limit):
    phase = event_log.get("phase", "")
    peak = event_log.get("peak_memory", 0)
    if phase == "build" or peak > workspace_limit:
        return "build"
    return "runtime"

def extract_exclusion_evidence(engine_logs):
    excluded = []
    for line in engine_logs:
        if "tactic" in line.lower() and "rejected" in line.lower():
            parts = line.split(":")
            excluded.append(parts[0].strip())
    return sorted(excluded)
