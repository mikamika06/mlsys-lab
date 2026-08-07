def run_needle_probe(model_runner, context, needle, query):
    full_text = context + " " + needle + " " + query
    response = model_runner(full_text)
    success = needle.strip().lower() in response.strip().lower()
    return {"success": bool(success), "response": response}
