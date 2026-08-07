import ref


def check(workdir):
    from aneplan.routing import recompute_routing
    export = ref.SAMPLE_EXPORT
    want = ref.recompute_routing(export)
    got = recompute_routing(export)
    match = 1.0 if got == want else 0.0
    out = {"routing_match": match}
    if match == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
