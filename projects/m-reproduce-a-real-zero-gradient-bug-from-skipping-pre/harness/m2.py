import ref

def check(workdir):
    from kbitbug.packing import compute_token_utilization
    lengths_list = [
        [50, 100, 30, 200, 40],
        [120, 15, 80, 90, 110, 20],
        [500, 10, 20]
    ]
    max_len = 256

    ok = 0
    for lengths in lengths_list:
        got = compute_token_utilization(lengths, max_len)
        want = ref.compute_token_utilization(lengths, max_len)
        if isinstance(got, dict) and abs(got.get("packing_utilization", 0) - want["packing_utilization"]) < 1e-5:
            ok += 1

    out = {"utilization_accuracy": 1.0 if ok == len(lengths_list) else 0.0}
    if out["utilization_accuracy"] == 0.0:
        out["_note"] = f"Passed {ok}/{len(lengths_list)} packing utilization test cases."
    return out
