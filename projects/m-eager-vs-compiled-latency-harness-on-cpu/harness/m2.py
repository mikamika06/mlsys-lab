import ref

def check(workdir):
    from cpuharness.safety import verify_fullgraph_capture
    out = {"exception_caught": 0.0, "fallback_handled": 0.0}
    model = ref.get_bad_model()
    inputs = ref.get_test_inputs()
    try:
        exc_type = verify_fullgraph_capture(model, inputs)
        if exc_type is not None and isinstance(exc_type, type):
            out["exception_caught"] = 1.0
            out["fallback_handled"] = 1.0
        else:
            out["_note"] = f"verify_fullgraph_capture did not return an exception type: {exc_type}"
    except Exception as e:
        out["_note"] = f"verify_fullgraph_capture raised unexpected exception {type(e).__name__}"
    return out
