import ref


def check(workdir):
    from ggufschema.validator import find_first_missing_key

    out = {"missing_keys_matched": 0.0}
    ok = 0
    for i, item in enumerate(ref.SCHEMAS):
        want = ref.find_first_missing_key(item["metadata"], item["required"]) if hasattr(ref, "find_first_missing_key") else None
        # Compute reference locally if needed
        def local_ref(meta, req):
            for k in req:
                if k not in meta:
                    return k
            return None
        w = local_ref(item["metadata"], item["required"])
        got = find_first_missing_key(item["metadata"], item["required"])
        if got == w:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"schema {i}: got {got}, want {w}"
    out["missing_keys_matched"] = float(ok)
    return out
