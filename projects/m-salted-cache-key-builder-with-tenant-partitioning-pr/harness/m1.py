import ref


def check(workdir):
    from cachekey.builder import build_prefix_keys

    out = {"keys_matched": 0.0, "scenarios": float(len(ref.TEST_SCENARIOS))}
    ok = 0
    for i, sc in enumerate(ref.TEST_SCENARIOS):
        want = ref.build_prefix_keys(
            sc["tenant_id"], sc["tokens"], sc["block_size"], salt=sc["salt"]
        )
        try:
            got = build_prefix_keys(
                sc["tenant_id"], sc["tokens"], sc["block_size"], salt=sc["salt"]
            )
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"scenario {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"scenario {i} raised {type(e).__name__}: {e}"
    out["keys_matched"] = float(ok)
    return out
