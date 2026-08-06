import ref


def check(workdir):
    out = {"opcheck_passed": 0.0}
    try:
        from customop.ops import register_custom_op, validate_op_schema
        register_custom_op()
        x, alpha = ref.make_test_data(42)
        passed = validate_op_schema(x, alpha)
        out["opcheck_passed"] = 1.0 if passed else 0.0
    except Exception as e:
        out["_note"] = f"Error checking milestone 1: {type(e).__name__}: {str(e)[:120]}"
    return out
