from sloclassify.parser import parse_request
from sloclassify.classify import classify_violation

def generate_report(requests, slo_target):
    counts = {"none": 0, "queueing": 0, "long-prefill": 0, "long-output": 0}
    for r in requests:
        parsed = parse_request(r)
        cause = classify_violation(parsed, slo_target)
        counts[cause] += 1
    return counts
