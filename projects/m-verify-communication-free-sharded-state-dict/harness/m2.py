import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    from fsdp_verify.wrap_order import analyze_wrap_violations

    out = {"wrap_violations_measured": 0.0, "total": float(len(ref.TREES_AND_WRAP_SEQUENCES))}
    ok = 0

    for i, (tree, seq) in enumerate(ref.TREES_AND_WRAP_SEQUENCES):
        want = ref.ref_analyze_wrap(tree, seq)
        try:
            got = analyze_wrap_violations(tree, seq)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {i}: got {got}, want {want}"
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}: {str(e)[:100]}"

    out["wrap_violations_measured"] = float(ok)
    return out
