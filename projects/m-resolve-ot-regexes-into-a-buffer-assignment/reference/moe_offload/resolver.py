import re

def resolve_ot_regexes(tensors, overrides):
    assignments = {}
    compiled = [(re.compile(pat), target) for pat, target in overrides]
    for t_name, t_info in sorted(tensors.items(), key=lambda x: x[0]):
        assigned_target = "default"
        for pat, target in compiled:
            if pat.search(t_name):
                assigned_target = target
                break
        assignments[t_name] = {
            "target": assigned_target,
            "size_bytes": t_info["size_bytes"],
            "is_moe": t_info.get("is_moe", False),
            "moe_layer": t_info.get("moe_layer", -1)
        }
    return assignments
