import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from onnxcalc.broadcast import compute_broadcast_shape
    except Exception as e:
        return {"shapes_matched": 0.0, "total": float(len(ref.TEST_SHAPES)), "_note": f"import failed: {e}"}

    out = {"shapes_matched": 0.0, "total": float(len(ref.TEST_SHAPES))}
    ok = 0
    for idx, item in enumerate(ref.TEST_SHAPES):
        s1, s2, want = item
        if want is ValueError or (isinstance(want, type) and issubclass(want, Exception)):
            try:
                compute_broadcast_shape(s1, s2)
                if "_note" not in out:
                    out["_note"] = f"test {idx}: expected ValueError for shapes {s1} and {s2}, but succeeded"
            except ValueError:
                ok += 1
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"test {idx}: expected ValueError, got {type(e).__name__}"
        else:
            try:
                got = compute_broadcast_shape(s1, s2)
                if list(got) == list(want):
                    ok += 1
                elif "_note" not in out:
                    out["_note"] = f"test {idx}: got {got}, want {want}"
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"test {idx}: raised {type(e).__name__}: {str(e)}"
    out["shapes_matched"] = float(ok)
    return out
