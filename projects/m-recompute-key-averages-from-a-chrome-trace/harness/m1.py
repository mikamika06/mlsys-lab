import ref
from tracex.parse import compute_key_averages


def check(workdir):
    t_a, _ = ref.generate_traces()
    want = compute_key_averages(t_a)
    from tracex.parse import compute_key_averages as learner_parse
    try:
        got = learner_parse(t_a)
    except Exception as e:
        return {"keys_matched": 0.0, "_note": f"raised exception: {e}"}
    matched = 0
    for k in want:
        if k in got and got[k].get("count") == want[k]["count"]:
            matched += 1
    return {"keys_matched": float(matched)}
