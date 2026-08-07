from long_ctx.analyzer import compare_extension_methods

def check(workdir):
    m = {"methods_compared": 0.0}
    try:
        res = compare_extension_methods(lambda x: "A", lambda x: "B", "test")
        if isinstance(res, dict) and "method_a" in res and "method_b" in res:
            m["methods_compared"] = 1.0
    except Exception:
        pass
    return m
