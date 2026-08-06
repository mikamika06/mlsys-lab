SAMPLE_LOGS = [
    {"is_recompile": True, "guard_id": 10},
    {"is_recompile": True, "guard_id": 11},
    {"is_recompile": True, "guard_id": 10},
    {"is_recompile": True, "guard_id": 12}
]

def oracle_count(logs):
    seen = set()
    c = 0
    for l in logs:
        if l.get("is_recompile"):
            gid = l.get("guard_id")
            if gid not in seen:
                seen.add(gid)
                c += 1
    return c
