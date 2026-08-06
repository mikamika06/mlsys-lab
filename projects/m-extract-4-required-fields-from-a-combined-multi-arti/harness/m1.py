import ref

def check(workdir):
    from extractor.parse import extract_required_fields
    log_data, _ = ref.generate_fixtures()
    expected = ref.extract_fields(log_data)
    try:
        actual = extract_required_fields(log_data)
    except Exception:
        return {"fields_matched": 0.0, "_note": "extraction raised an exception"}

    if actual == expected:
        return {"fields_matched": 1.0}
    else:
        return {"fields_matched": 0.0, "_note": f"got {actual}, expected {expected}"}
