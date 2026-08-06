from imatrix.core import verify_tensor_shape

def validate_imatrix(model_tensors, imatrix_data):
    mismatches = []
    missing = []
    extra = []
    model_keys = set(model_tensors.keys())
    imatrix_keys = set(imatrix_data.keys())

    for k in sorted(model_keys - imatrix_keys):
        missing.append(k)
    for k in sorted(imatrix_keys - model_keys):
        extra.append(k)

    for k in sorted(model_keys & imatrix_keys):
        ok, msg = verify_tensor_shape(k, model_tensors[k], imatrix_data[k])
        if not ok:
            mismatches.append({"tensor": k, "error": msg})

    return {
        "valid": len(mismatches) == 0 and len(missing) == 0,
        "mismatches": mismatches,
        "missing": missing,
        "extra": extra
    }
