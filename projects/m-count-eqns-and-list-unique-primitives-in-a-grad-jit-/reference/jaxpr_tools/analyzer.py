def count_equations(jaxpr):
    count = len(jaxpr.eqns)
    for eqn in jaxpr.eqns:
        for subjax in eqn.params.get("call_jaxprs", []):
            count += count_equations(subjax)
        if "jaxpr" in eqn.params:
            count += count_equations(eqn.params["jaxpr"])
    return count


def list_unique_primitives(jaxpr):
    prims = set()
    for eqn in jaxpr.eqns:
        prims.add(str(eqn.primitive.name))
        for subjax in eqn.params.get("call_jaxprs", []):
            prims.update(list_unique_primitives(subjax))
        if "jaxpr" in eqn.params:
            prims.update(list_unique_primitives(eqn.params["jaxpr"]))
    return sorted(list(prims))
