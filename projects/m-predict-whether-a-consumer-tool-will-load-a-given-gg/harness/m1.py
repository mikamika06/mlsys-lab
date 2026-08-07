import ref


def check(workdir):
    from gguf_interop.compat import check_tool_compatibility

    out = {"cases_matched": 0.0}
    matched = 0

    for meta in ref.TEST_METADATA:
        for profile in ref.TOOL_PROFILES:
            want = ref.oracle_check_tool_compatibility(meta, profile)
            got = check_tool_compatibility(meta, profile)

            if isinstance(got, dict) and got.get("compatible") == want.get("compatible"):
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"mismatch on {meta.get('general.architecture')} / {profile.get('name')}: got {got}, want {want}"

    out["cases_matched"] = float(matched)
    return out
