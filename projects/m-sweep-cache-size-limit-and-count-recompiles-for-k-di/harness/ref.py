import re

SHAPE_DATASETS = [
    [(1, 64), (2, 64), (3, 64), (4, 64), (5, 64), (1, 64), (6, 64)],
    [(16, 128), (32, 128), (16, 128), (48, 128), (64, 128), (80, 128)],
    [(8, 8), (8, 16), (8, 24), (8, 32)]
]

CACHE_LIMIT_SWEEPS = [
    [1, 2, 4, 8],
    [2, 3, 5],
    [1, 10]
]

SAMPLE_LOGS = """
[GUARDS] TREE_GUARD: L['x'].size()[0] == 16
[GUARDS] TREE_GUARD: L['x'].size()[1] == 128
[GUARDS] TREE_GUARD: L['y'].size()[0] == 32
[GUARDS] IGNORED: float_arg == 1.0
[GUARDS] TREE_GUARD: L['y'].size()[1] == 128
"""

def sweep_cache_limit(shapes, cache_size_limits):
    results = []
    for limit in cache_size_limits:
        seen = set()
        recompiles = 0
        fallbacks = 0
        for s in shapes:
            if s in seen:
                continue
            if len(seen) < limit:
                recompiles += 1
                seen.add(s)
            else:
                fallbacks += 1
        results.append({
            "cache_size_limit": limit,
            "recompiles": recompiles,
            "eager_fallbacks": fallbacks,
            "cache_exhausted": len(seen) >= limit and fallbacks > 0
        })
    return results

def extract_shape_guards(guard_logs):
    guards = []
    pattern = re.compile(r"L\['(\w+)'\]\.size\(\)\[(\d+)\]\s*==\s*(\d+)")
    for line in guard_logs.splitlines():
        if "GUARD" in line or "size()" in line:
            matches = pattern.findall(line)
            for var, dim, val in matches:
                expr = f"L['{var}'].size()[{dim}] == {val}"
                guards.append({
                    "var": var,
                    "dim": int(dim),
                    "val": int(val),
                    "expr": expr
                })
    return guards

def simulate_recompile_storm(shapes, cache_size_limit):
    compiled_shapes = set()
    history = []
    fallback_step = None
    for idx, shape in enumerate(shapes):
        if shape in compiled_shapes:
            status = "hit"
        elif len(compiled_shapes) < cache_size_limit:
            status = "recompile"
            compiled_shapes.add(shape)
        else:
            status = "eager_fallback"
            if fallback_step is None:
                fallback_step = idx
        history.append({"step": idx, "shape": shape, "status": status})
    return {
        "history": history,
        "fallback_step": fallback_step,
        "total_recompiles": len(compiled_shapes),
        "total_fallbacks": sum(1 for h in history if h["status"] == "eager_fallback")
    }
