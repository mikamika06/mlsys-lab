import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    from export_verify.verifier import inspect_strict_export_behavior, verify_roundtrip_equivalence

    out = {"roundtrip_match": 0.0, "strict_inspection_match": 0.0}

    mod, ep, args = ref.get_standard_test_case()
    eq = verify_roundtrip_equivalence(ep, args)
    if not eq:
        out["_note"] = "verify_roundtrip_equivalence failed on standard exported program"
        return out
    out["roundtrip_match"] = 1.0

    mut_mod, mut_args = ref.get_mutating_test_case()
    inspection = inspect_strict_export_behavior(mut_mod, mut_args)

    if not isinstance(inspection, dict):
        out["_note"] = "inspect_strict_export_behavior did not return a dict"
        return out

    required_keys = {
        "strict_success",
        "strict_error",
        "nonstrict_success",
        "nonstrict_error",
        "strict_has_mutations",
        "nonstrict_has_mutations",
    }
    if not required_keys.issubset(inspection.keys()):
        out["_note"] = f"Missing keys in inspection result: {required_keys - set(inspection.keys())}"
        return out

    if inspection["nonstrict_success"] is not True:
        out["_note"] = f"Expected nonstrict_success=True, got {inspection}"
        return out

    out["strict_inspection_match"] = 1.0
    return out
