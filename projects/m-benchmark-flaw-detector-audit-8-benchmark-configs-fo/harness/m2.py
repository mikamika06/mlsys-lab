import ref


def check(workdir):
    from benchaudit.reconstruct import reconstruct_cli

    ok = True
    for meta in ref.RESULT_METADATA:
        want = ref.reconstruct_cli(meta)
        got = reconstruct_cli(meta)
        if got != want:
            ok = False
            break

    if ok:
        return {"reconstructions_matched": 1.0}
    return {"reconstructions_matched": 0.0, "_note": "Reconstructed CLI command did not match reference"}
