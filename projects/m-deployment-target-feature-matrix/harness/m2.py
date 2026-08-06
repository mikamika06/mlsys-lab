import ref

def check(workdir):
    from exporttools.repair import repair_image_input
    out = {"repair_matched": 0.0}
    inputs = [
        {},
        {"scale": 0.5},
        {"color_space": "BGR"}
    ]
    ok = True
    for inp in inputs:
        want = ref.repair_image_input(inp)
        got = repair_image_input(inp)
        if got != want:
            ok = False
            break
    if ok:
        out["repair_matched"] = 1.0
    return out
