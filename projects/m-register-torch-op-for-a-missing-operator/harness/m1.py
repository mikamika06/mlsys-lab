import ref

def check(workdir):
    from milpass import register
    out = {"reg_matched": 0.0}
    try:
        res = register.register_torch_op()
        want = ref.expected_registered_op()
        if isinstance(res, dict) and res.get("name") == want["name"] and res.get("schema") == want["schema"]:
            out["reg_matched"] = 1.0
        else:
            out["_note"] = f"got {res}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {str(e)}"
    return out
