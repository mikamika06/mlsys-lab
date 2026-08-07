import ref

def check(workdir):
    from blobgraph.manifest import parse_manifest

    out = {"manifests_matched": 0.0}
    ok = 0
    for i, m_str in enumerate(ref.SAMPLE_MANIFESTS):
        want = ref.parse_manifest(m_str)
        try:
            got = parse_manifest(m_str)
        except Exception as e:
            out["_note"] = f"manifest {i} raised {type(e).__name__}: {str(e)[:100]}"
            return out
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"manifest {i}: got {got}, want {want}"
    out["manifests_matched"] = float(ok)
    return out
