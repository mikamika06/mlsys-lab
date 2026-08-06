import ref


def check(workdir):
    from quantmap.audit import audit_dependencies
    from quantmap.rule import build_bpp_table

    out = {"table_matched": 0.0, "audit_matched": 0.0}
    ok_table = 0
    ok_audit = 0

    for i, cfg in enumerate(ref.CONFIGS):
        want_table = ref.build_bpp_table(cfg["model"], cfg["libs"])
        got_table = build_bpp_table(cfg["model"], cfg["libs"])
        if got_table == want_table:
            ok_table += 1
        elif "_note" not in out:
            out["_note"] = f"table config {i}: got {got_table}, expected {want_table}"

        want_audit = ref.audit_dependencies(cfg["manifest"], ref.REGISTRY)
        got_audit = audit_dependencies(cfg["manifest"], ref.REGISTRY)
        if got_audit == want_audit:
            ok_audit += 1
        elif "_note" not in out:
            out["_note"] = f"audit config {i}: got {got_audit}, expected {want_audit}"

    out["table_matched"] = float(ok_table)
    out["audit_matched"] = float(ok_audit)
    return out
