def check(workdir):
    from runner import adapter

    m = {"tools_ok": 0.0}
    try:
        resp = {"function_call": {"name": "test"}}
        fixed = adapter.fix_tool_calling(resp)
        if isinstance(fixed, dict) and "tool_calls" in fixed:
            m["tools_ok"] = 1.0
    except Exception:
        pass
    return m
