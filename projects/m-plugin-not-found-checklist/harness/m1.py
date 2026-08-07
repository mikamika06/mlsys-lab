import ref


def check(workdir):
    from trtplugin.checklist import diagnose_plugin_issue

    out = {"diagnostics_matched": 0.0}
    total = len(ref.TEST_CHECKLIST_CASES)
    matched = 0

    for case in ref.TEST_CHECKLIST_CASES:
        res = diagnose_plugin_issue(case["requested"], case["registered"])
        if isinstance(res, dict) and res.get("status") == case["expected"]:
            matched += 1
        else:
            if "_note" not in out:
                out["_note"] = f"Failed for {case['requested']['name']}: got {res}, expected status {case['expected']}"

    if matched == total:
        out["diagnostics_matched"] = 1.0
    return out
