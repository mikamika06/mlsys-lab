import ref


def check(workdir):
    from migration.artifact import roundtrip_export_load

    spec = ref.generate_spec(42)
    try:
        res = roundtrip_export_load(spec)
        match = 1 if res else 0
    except Exception as e:
        return {"roundtrip_match": 0, "_note": f"raised {type(e).__name__}"}
    return {"roundtrip_match": match}
