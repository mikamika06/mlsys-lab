def count_equations(jaxpr_data):
    if "eqns" in jaxpr_data:
        return len(jaxpr_data["eqns"])
    return 0


def list_unique_primitives(jaxpr_data):
    prims = set()
    eqns = jaxpr_data.get("eqns", [])
    for eqn in eqns:
        if "primitive" in eqn:
            prims.add(eqn["primitive"])
    return sorted(list(prims))


def safe_trace_collector(fn, sample_inputs):
    cache = []
    for x in sample_inputs:
        res = fn(x)
        cache.append(res)
    return cache
