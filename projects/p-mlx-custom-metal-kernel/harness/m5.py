def check(workdir):
    import ref
    m = {"integration_ok": 0.0}
    try:
        from metal_op.model import FusedModel
        model = FusedModel()
        inputs = ref.get_test_inputs()
        out = model.forward(inputs[0])
        if out is not None and len(out) == len(inputs[0]):
            m["integration_ok"] = 1.0
    except Exception:
        pass
    return m
