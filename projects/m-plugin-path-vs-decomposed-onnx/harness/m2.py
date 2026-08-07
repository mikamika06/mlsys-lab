import ref

def check(workdir):
    from trtplug.migration import V2ToV3Adapter

    out = {"methods_matched": 0.0, "wrapper_valid": 0.0}
    try:
        mock_v2 = ref.MockV2Plugin({"alpha": 0.5})
        adapter = V2ToV3Adapter(mock_v2)
        fields = adapter.get_field_values()
        fmt = adapter.supports_format_combination(0, 0, 0)
        if fields == {"alpha": 0.5}:
            out["methods_matched"] = 1.0
        if fmt is True:
            out["wrapper_valid"] = 1.0
    except Exception as e:
        out["_note"] = str(e)[:120]
    return out
