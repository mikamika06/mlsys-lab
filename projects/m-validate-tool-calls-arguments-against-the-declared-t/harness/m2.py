import ref


def check(workdir):
    out = {"format_json_failures_detected": 0.0, "roundtrip_matches": 0.0}
    try:
        from tool_val.schema import roundtrip_schema
        from tool_val.validator import demonstrate_format_json_non_conformance
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {e}"
        return out

    try:
        got_demo = demonstrate_format_json_non_conformance(
            ref.RAW_RESPONSES, ref.USER_SCHEMA
        )
    except Exception as e:
        out["_note"] = f"demonstrate_format_json_non_conformance raised: {e}"
        return out

    ref_demo = ref.reference_demonstrate_format_json_non_conformance(
        ref.RAW_RESPONSES, ref.USER_SCHEMA
    )

    demo_ok = True
    if len(got_demo) != len(ref_demo):
        demo_ok = False
        out["_note"] = "demonstrate_format_json_non_conformance length mismatch"
    else:
        for g, r in zip(got_demo, ref_demo):
            if (
                g.get("id") != r.get("id")
                or g.get("is_valid_json") != r.get("is_valid_json")
                or g.get("is_schema_valid") != r.get("is_schema_valid")
            ):
                demo_ok = False
                out["_note"] = f"Mismatch in response evaluation for id={r.get('id')}"
                break

    if demo_ok:
        out["format_json_failures_detected"] = 1.0

    roundtrip_ok = True
    for name, schema in ref.TOOL_SCHEMAS.items():
        try:
            got_rt = roundtrip_schema(schema)
        except Exception as e:
            roundtrip_ok = False
            out["_note"] = f"roundtrip_schema raised on {name}: {e}"
            break

        ref_rt = ref.reference_roundtrip_schema(schema)
        if got_rt != ref_rt:
            roundtrip_ok = False
            out["_note"] = f"roundtrip_schema mismatch for schema {name}"
            break

    if roundtrip_ok:
        out["roundtrip_matches"] = 1.0

    return out
