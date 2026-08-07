import ref


def check(workdir):
    from gguf_quant.validator import validate_pruned_model

    model = ref.get_sample_model()
    pruned = [1]
    valid = validate_pruned_model(model, pruned)

    out = {"pruned_valid": 0.0}
    if valid is True:
        out["pruned_valid"] = 1.0
    else:
        out["_note"] = f"validate_pruned_model returned {valid}"
    return out
