import ref


def check(workdir):
    from dyncomp.backends import find_experimental_backends
    got = find_experimental_backends()
    want = ref.ref_find_experimental_backends()

    out = {"experimental_matched": 0.0}
    if sorted(got) == sorted(want):
        out["experimental_matched"] = 1.0
    else:
        out["_note"] = f"Expected experimental backends {sorted(want)}, got {sorted(got)}"
    return out
