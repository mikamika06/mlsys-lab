import ref


def check(workdir):
    from quantizer.opset import identify_intermediate_opset, validate_opset_compatibility

    matched = 0.0
    for case in ref.CASES:
        want = ref.identify_intermediate_opset(case["nodes"], case["graph_opset_map"])
        got = identify_intermediate_opset(case["nodes"], case["graph_opset_map"])

        want_compat = ref.validate_opset_compatibility(want)
        got_compat = validate_opset_compatibility(got)

        if got == want and got_compat == want_compat:
            matched += 1.0

    return {"opsets_matched": matched}
