import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    from fsdp_verify.state_dict import verify_communication_free_state_dict

    out = {"state_dicts_verified": 0.0, "total": float(len(ref.STATE_DICT_CASES))}
    ok = 0

    for i, (specs, ws, rank) in enumerate(ref.STATE_DICT_CASES):
        want = ref.ref_verify_state_dict(specs, ws, rank)
        try:
            got = verify_communication_free_state_dict(specs, ws, rank)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {i}: got {got}, want {want}"
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}: {str(e)[:100]}"

    out["state_dicts_verified"] = float(ok)
    return out
