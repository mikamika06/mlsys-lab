import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    from export_verify.verifier import verify_graph_signature

    out = {"signature_matches": 0.0}

    mod, ep, args = ref.get_standard_test_case()
    valid, details = verify_graph_signature(mod, ep, args)

    if not valid:
        out["_note"] = f"Expected valid=True for matching export, got {valid}, details={details}"
        return out

    if not details.get("params_ok") or not details.get("buffers_ok"):
        out["_note"] = "params_ok or buffers_ok is False for matching export"
        return out

    mod_mismatch, ep_mismatch, args_mismatch = ref.get_standard_test_case()
    mod_mismatch.weight = torch_param = ref.torch.nn.Parameter(ref.torch.randn(8, 16))
    valid_bad, details_bad = verify_graph_signature(mod_mismatch, ep_mismatch, args_mismatch)

    if valid_bad:
        out["_note"] = "verify_graph_signature returned True for mismatched parameter shape"
        return out

    if details_bad.get("param_shapes_ok") is not False:
        out["_note"] = f"expected param_shapes_ok=False, got {details_bad}"
        return out

    out["signature_matches"] = 1.0
    return out
