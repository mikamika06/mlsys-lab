import json

def classify_failures(logs: list[str]) -> dict[str, int]:
    counts = {"valid": 0, "extra_text": 0, "truncated": 0, "type_error": 0}
    for text in logs:
        text = text.strip()
        if not text.startswith("{"):
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                try:
                    json.loads(text[start:end])
                    counts["extra_text"] += 1
                except json.JSONDecodeError:
                    counts["truncated"] += 1
            else:
                counts["truncated"] += 1
            continue

        try:
            obj = json.loads(text)
            if isinstance(obj.get("age"), str):
                counts["type_error"] += 1
            else:
                counts["valid"] += 1
        except json.JSONDecodeError:
            counts["truncated"] += 1

    return counts
