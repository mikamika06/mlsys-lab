import ref


def check(workdir):
    from templater import tools
    out = {"tool_call_valid": 0.0}
    sample_output = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
    ok = 0
    for case in ref.TEST_CASES:
        want = ref.render_and_validate_tool(case, sample_output)
        got = tools.render_and_validate_tool(case, sample_output)
        if got == want:
            ok += 1
    if ok == len(ref.TEST_CASES):
        out["tool_call_valid"] = 1.0
    return out
