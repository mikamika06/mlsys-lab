import ref

def check(workdir):
    from gguf_tool.validate import validate_filename
    out = {"filenames_matched": 0.0}
    ok = True
    for fn in ref.VALID_FILENAMES:
        if not validate_filename(fn):
            ok = False
            out["_note"] = f"Valid filename {fn} rejected"
            break
    if ok:
        for fn in ref.INVALID_FILENAMES:
            if validate_filename(fn):
                ok = False
                out["_note"] = f"Invalid filename {fn} accepted"
                break
    if ok:
        out["filenames_matched"] = 1.0
    return out
