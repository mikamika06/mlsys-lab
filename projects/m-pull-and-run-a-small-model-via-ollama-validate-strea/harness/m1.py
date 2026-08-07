import ref

def check(workdir):
    from ollamautil.stream import parse_stream
    out = {"stream_valid": 0.0, "schema_matched": 0.0}
    test_chunks = [
        '{"status": "ok", "value": 42}',
        '{"status": "running", "value": 100}'
    ]
    try:
        results = parse_stream(test_chunks, ref.SAMPLE_SCHEMA)
        if isinstance(results, list) and len(results) == len(test_chunks):
            out["stream_valid"] = 1.0
            if all(ref.validate_stream_chunk(c, ref.SAMPLE_SCHEMA) for c in results):
                out["schema_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"m1 failed: {type(e).__name__}: {str(e)[:100]}"
    return out
