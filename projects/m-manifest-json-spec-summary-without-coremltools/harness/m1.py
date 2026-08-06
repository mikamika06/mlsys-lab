import ref


def check(workdir):
    from mlpackage.manifest import summarize_manifest
    tmp = ref.create_mock_package()
    try:
        want = ref.ref_summarize(tmp.name)
        got = summarize_manifest(tmp.name)
        if got == want:
            return {"manifest_match": 1.0}
        return {"manifest_match": 0.0, "_note": f"got {got}, want {want}"}
    finally:
        tmp.cleanup()
