import json

def classify_failure(raw_output: str, schema: dict) -> str:
    s = raw_output.strip()
    if s.startswith("```") or "Here is" in raw_output or raw_output.startswith("\n"):
        if not s.startswith("{"):
            return "extra_text"

    try:
        data = json.loads(s)
    except Exception:
        if s.startswith("{") and not s.endswith("}"):
            return "truncated"
        return "extra_text"

    req_fields = schema.get("required", [])
    properties = schema.get("properties", {})

    for f in req_fields:
        if f not in data:
            return "schema_mismatch"

    for k, v in data.items():
        if k in properties:
            expected_type = properties[k].get("type")
            if expected_type == "integer" and not isinstance(v, int):
                return "schema_mismatch"
            if expected_type == "string" and not isinstance(v, str):
                return "schema_mismatch"
            if expected_type == "array" and not isinstance(v, list):
                return "schema_mismatch"

    return "valid"

def summarize_failures(failures: list) -> dict:
    summary = {"extra_text": 0, "truncated": 0, "schema_mismatch": 0, "valid": 0}
    for f in failures:
        cat = classify_failure(f["output"], f["schema"])
        summary[cat] = summary.get(cat, 0) + 1
    return summary
