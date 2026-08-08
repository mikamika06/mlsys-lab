def import_or_catch_error(model_spec):
    unsupported = model_spec.get("unsupported_op")
    if unsupported:
        return {"success": False, "error_type": "UnsupportedError", "op": unsupported}
    return {"success": True, "error_type": None, "op": None}
