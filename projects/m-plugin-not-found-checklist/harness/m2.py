import ref


def check(workdir):
    from trtplugin.serialize import serialize_fields, deserialize_fields
    from trtplugin.decision import decide_op_strategy

    out = {"serialize_matched": 0.0, "decisions_matched": 0.0}

    s_ok = True
    for payload in ref.TEST_SERIALIZE_PAYLOADS:
        try:
            buf = serialize_fields(payload)
            rec = deserialize_fields(buf)
            if rec != payload:
                s_ok = False
                out["_note"] = f"Serialization roundtrip mismatched: got {rec}, expected {payload}"
                break
        except Exception as e:
            s_ok = False
            out["_note"] = f"Serialization exception: {type(e).__name__}: {str(e)}"
            break

    if s_ok:
        out["serialize_matched"] = 1.0

    d_ok = True
    for case in ref.TEST_DECISION_CASES:
        try:
            res = decide_op_strategy(case["node"], case["trt_native"], case["plugins"], case["constraints"])
            if res != case["expected"]:
                d_ok = False
                out["_note"] = f"Decision mismatch for {case['node']['op_type']}: got {res}, expected {case['expected']}"
                break
        except Exception as e:
            d_ok = False
            out["_note"] = f"Decision exception: {type(e).__name__}: {str(e)}"
            break

    if d_ok:
        out["decisions_matched"] = 1.0

    return out
