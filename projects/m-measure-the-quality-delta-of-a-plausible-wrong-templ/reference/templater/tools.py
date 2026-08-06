import json

def render_and_validate_tool(case, model_output):
    rendered = case["good_template"].format(prompt=case["prompt"], tools=json.dumps(case["tools"]))
    if not rendered:
        return False
    try:
        parsed = json.loads(model_output)
        return isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed
    except Exception:
        return False
