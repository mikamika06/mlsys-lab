import sys
sys.path.insert(0, ".")
from cacheplan.layout import build_turn_prompt

def test_dynamic_after_history():
    sys_b = ["s1", "s2"]
    hist_b = ["h1", "h2"]
    dyn_b = ["d1"]

    prompt = build_turn_prompt(sys_b, hist_b, dyn_b)

    hist_idx = prompt.index("h2")
    dyn_idx = prompt.index("d1")

    assert dyn_idx > hist_idx, "Dynamic blocks must be placed after history to avoid breaking the cache prefix"
