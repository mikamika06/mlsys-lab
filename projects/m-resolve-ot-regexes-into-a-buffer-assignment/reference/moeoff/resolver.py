import re

def resolve_ot_regexes(tensors, overrides, default_buffer="GPU"):
    assignments = {}
    for name, size in tensors:
        assigned = default_buffer
        for pattern, buf in overrides:
            if re.search(pattern, name):
                assigned = buf
        assignments[name] = assigned
    return assignments
