import ref

def check(workdir):
    from kvslot.sizing import calculate_slot_sizing
    v = 1024 * 1024 * 1024
    ns = 4
    cl = 512
    hs = 768
    nl = 12
    got = calculate_slot_sizing(v, ns, cl, hs, nl)
    want = ref.compute_slot_sizing(v, ns, cl, hs, nl)
    out = {"sizing_matched": 1.0 if got == want else 0.0}
    return out
