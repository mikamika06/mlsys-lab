import sys

sys.path.insert(0, ".")
from streammetrics.classify import classify_workload_dominance


def test_prefill_vs_decode_dominance():
    dom1 = classify_workload_dominance(ttft=2.0, prefill_tok_per_sec=500.0, decode_tok_per_sec=20.0, prompt_tokens=1000, completion_tokens=10)
    assert dom1 == "prefill-dominated", f"expected prefill-dominated, got {dom1}"

    dom2 = classify_workload_dominance(ttft=0.1, prefill_tok_per_sec=1000.0, decode_tok_per_sec=50.0, prompt_tokens=100, completion_tokens=200)
    assert dom2 == "decode-dominated", f"expected decode-dominated, got {dom2}"
