import json

TEST_CASES = [
    {
        "prompt": "What is the weather in Tokyo?",
        "tools": [{"name": "get_weather", "parameters": {"location": "string"}}],
        "good_template": "USER: {prompt}\nTOOLS: {tools}\nASSISTANT:",
        "wrong_template": "SYSTEM: do stuff\nUSER: {prompt}\nASSISTANT:",
        "stop_sequence": "<|end|>"
    },
    {
        "prompt": "Calculate 2 + 2",
        "tools": [{"name": "calculator", "parameters": {"expr": "string"}}],
        "good_template": "### User:\n{prompt}\n### Tools:\n{tools}\n### Response:\n",
        "wrong_template": "### User:\n{prompt}\n### Response:\n",
        "stop_sequence": "</s>"
    }
]

def compute_quality_delta(case):
    good_rendered = case["good_template"].format(prompt=case["prompt"], tools=json.dumps(case["tools"]))
    wrong_rendered = case["wrong_template"].format(prompt=case["prompt"], tools=json.dumps(case["tools"]))
    score_good = len(good_rendered) * 1.5
    score_wrong = len(wrong_rendered) * 0.8
    return abs(score_good - score_wrong)

def render_and_validate_tool(case, model_output):
    rendered = case["good_template"].format(prompt=case["prompt"], tools=json.dumps(case["tools"]))
    try:
        parsed = json.loads(model_output)
        return "name" in parsed and "arguments" in parsed
    except Exception:
        return False

def check_stop_sequences(template, stop_seq, sample_text):
    if not stop_seq:
        return False
    return stop_seq in sample_text
