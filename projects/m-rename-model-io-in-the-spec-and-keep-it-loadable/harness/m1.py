import ref


def check(workdir):
    from mlspec.io import rename_io
    ok = True
    for spec, imap, omap in ref.SPECS:
        got = rename_io(spec, imap, omap)
        want = ref.rename_io(spec, imap, omap)
        if got != want:
            ok = False
    return {"io_matched": 1.0 if ok else 0.0}
