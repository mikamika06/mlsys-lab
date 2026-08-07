import ref

def check(workdir):
    from quantexport.costs import requant_cost_share
    out = {"cost_share_matched": 0.0}
    ok = 0
    for nodes, total in ref.CASES_COST:
        want = ref.ref_requant_cost_share(nodes, total)
        got = requant_cost_share(nodes, total)
        if abs(got - want) < 1e-5:
            ok += 1
    out["cost_share_matched"] = float(ok)
    return out
