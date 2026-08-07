import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from polyiso.isolation import build_isolation_command, build_mark_all_command

    out = {"commands_matched": 0.0}
    ok = 0
    total = len(ref.TEST_CASES_COMMANDS) + 1

    for tc in ref.TEST_CASES_COMMANDS:
        want_iso = ref.build_isolation_command(tc["onnx"], tc["trt"], tc["layers"], tc["out_json"])
        got_iso = build_isolation_command(tc["onnx"], tc["trt"], tc["layers"], tc["out_json"])
        if got_iso == want_iso:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"iso cmd mismatch: got {got_iso}, want {want_iso}"

    want_mark = ref.build_mark_all_command("model.onnx", "marked.onnx")
    got_mark = build_mark_all_command("model.onnx", "marked.onnx")
    if got_mark == want_mark:
        ok += 1
    elif "_note" not in out:
        out["_note"] = f"mark cmd mismatch: got {got_mark}, want {want_mark}"

    if ok == total:
        out["commands_matched"] = 1.0
    return out
