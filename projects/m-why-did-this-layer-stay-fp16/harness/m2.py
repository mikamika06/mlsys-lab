import ref

def check(workdir):
    from quantopt.search import find_budget_config

    layers = ref.make_layers()
    max_bytes = 4096
    want = ref.search_budget(layers, max_bytes)
    got = find_budget_config(layers, max_bytes)
    out = {"budget_matched": 1.0 if got == want else 0.0}
    if got != want:
        out["_note"] = f"got {got}, want {want}"
    return out
