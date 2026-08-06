from profilediag.classifier import classify_error

def triage_transcripts(transcripts):
    results = []
    for item in transcripts:
        log_text = item.get("log", "")
        env_info = item.get("env", {})
        category = classify_error(log_text, env_info)
        results.append({
            "id": item.get("id"),
            "category": category
        })
    return results
