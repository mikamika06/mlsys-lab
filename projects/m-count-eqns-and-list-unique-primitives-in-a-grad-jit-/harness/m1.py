import ref


def check(workdir):
    from jaxpr_tools.analyzer import count_equations, list_unique_primitives

    jaxpr = ref.make_sample_jaxpr()

    got_count = count_equations(jaxpr)
    want_count = ref.count_equations(jaxpr) if hasattr(ref, "count_equations") else len(jaxpr.eqns)

    got_prims = list_unique_primitives(jaxpr)
    want_prims = ref.list_unique_primitives(jaxpr) if hasattr(ref, "list_unique_primitives") else sorted(list({str(e.primitive.name) for e in jaxpr.eqns}))

    out = {"metrics_matched": 0.0}
    matches = 0
    if got_count == want_count:
        matches += 1
    if sorted(got_prims) == sorted(want_prims):
        matches += 1
    if isinstance(got_prims, list):
        matches += 1

    out["metrics_matched"] = float(matches)
    return out
