import ref

def check(workdir):
    from app.analyzer import classify_failures
    logs = [
        '{"name": "john", "age": 25}',
        '{"name": "alice", "age": 30}',
        'Here is the JSON:\n{"name": "john", "age": 25}',
        '{"name": "john", "age":',
        '{"name": "alice", "age": "25"}'
    ]
    res = classify_failures(logs)
    return {
        "valid_ok": 1.0 if res.get("valid") == 2 else 0.0,
        "extra_ok": 1.0 if res.get("extra_text") == 1 else 0.0,
        "trunc_ok": 1.0 if res.get("truncated") == 1 else 0.0,
        "type_ok": 1.0 if res.get("type_error") == 1 else 0.0,
    }
