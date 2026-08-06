import ref


def check(workdir):
    from irhist.agreement import verify_numeric_agreement
    out = {"agreement_score": 0.0}
    direct, onnx = ref.generate_outputs(42)
    try:
        res = verify_numeric_agreement(direct, onnx, rtol=1e-5, atol=1e-5)
        if res is True:
            out["agreement_score"] = 1.0
    except Exception:
        pass
    return out
