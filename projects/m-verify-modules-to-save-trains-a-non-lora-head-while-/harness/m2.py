import ref


def check(workdir):
    out = {"ratio_matched": 0.0}
    try:
        from peft_verify.size import compute_adapter_size_ratio

        model = ref.build_test_model()
        adapter_dict = ref.build_adapter_dict()

        got = compute_adapter_size_ratio(model, adapter_dict)

        full_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        adapter_bytes = sum(t.numel() * t.element_size() for t in adapter_dict.values())
        expected = adapter_bytes / full_bytes

        if abs(got - expected) < 1e-6:
            out["ratio_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, expected {expected}"
    except Exception as e:
        out["_note"] = f"exception: {type(e).__name__}: {str(e)[:120]}"
    return out
