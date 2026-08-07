import ref


def check(workdir):
    from edgeml.modelfile import parse_modelfile, verify_modelfile
    out = {"modelfile_verified": 0.0}
    ok = 0
    for i, case in enumerate(ref.MODELFILE_CASES):
        parsed = parse_modelfile(case["text"])
        verified = verify_modelfile(parsed, case["expected_system"], case["expected_quant"])
        want_verified = ref.verify_modelfile_ref(
            ref.parse_modelfile_ref(case["text"]),
            case["expected_system"],
            case["expected_quant"]
        )
        if verified == want_verified and verified is True:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: verified={verified}, want={want_verified}"
    if ok == len(ref.MODELFILE_CASES):
        out["modelfile_verified"] = 1.0
    return out
