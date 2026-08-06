import json

def compute_quality_delta(case):
    good_rendered = case["good_template"].format(prompt=case["prompt"], tools=json.dumps(case["tools"]))
    wrong_rendered = case["wrong_template"].format(prompt=case["prompt"], tools=json.dumps(case["tools"]))
    score_good = len(good_rendered) * 1.5
    score_wrong = len(wrong_rendered) * 0.8
    return abs(score_good - score_wrong)
