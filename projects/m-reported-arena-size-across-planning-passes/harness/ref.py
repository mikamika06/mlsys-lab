import random

PASSES_LIST = [
    [{"arena_size": 2048}, {"arena_size": 1024}, {"arena_size": 1024}],
    [{"arena_size": 4096}, {"arena_size": 2048}, {"arena_size": 2048}, {"arena_size": 2048}],
    [{"arena_size": 512}, {"arena_size": 512}],
]

TENSORS_LIST = [
    [{"size": 1024, "is_constant": True}, {"size": 2048, "is_constant": False}],
    [{"size": 512, "is_constant": True}, {"size": 512, "is_constant": True}],
]

PROGRAMS_LIST = [
    {"nodes": [{"outputs": [{"name": "a", "shape": [1, 3, 224, 224]}]}]},
    {"nodes": [{"outputs": [{"name": "b", "shape": [1, "batch"]}]}]},
]

def track_arena_sizes(passes):
    sizes = []
    for p in passes:
        sizes.append(int(p.get("arena_size", 0)))
    converged = len(set(sizes[-2:])) == 1 if len(sizes) >= 2 else True
    return {"sizes": sizes, "converged": converged, "max_size": max(sizes) if sizes else 0}

def compute_separation_savings(tensors, method_a, method_b):
    total_bytes = sum(t["size"] for t in tensors)
    const_bytes = sum(t["size"] for t in tensors if t.get("is_constant", False))

    if method_a == "inline":
        cost_a = total_bytes
    elif method_a == "segmented":
        cost_a = total_bytes - const_bytes + (const_bytes // 2)
    else:
        cost_a = total_bytes

    if method_b == "inline":
        cost_b = total_bytes
    elif method_b == "segmented":
        cost_b = total_bytes - const_bytes + (const_bytes // 2)
    elif method_b == "isolated":
        cost_b = total_bytes - const_bytes
    else:
        cost_b = total_bytes

    return {
        "cost_a": cost_a,
        "cost_b": cost_b,
        "savings_a": total_bytes - cost_a,
        "savings_b": total_bytes - cost_b
    }

def detect_dynamic_tensors(program):
    unplanned = []
    for node in program.get("nodes", []):
        for out in node.get("outputs", []):
            shape = out.get("shape", [])
            if any(isinstance(dim, str) or dim is None or dim < 0 for dim in shape):
                unplanned.append(out.get("name", "unknown"))
    return unplanned
