import ref

def check(workdir):
    from glooprof.fingerprint import fingerprint_trace
    out = {"accuracy": 0.0}
    correct = 0
    for t in ref.TRACES:
        want = t["type"]
        got = fingerprint_trace(t)
        if got == want:
            correct += 1
    out["accuracy"] = float(correct / len(ref.TRACES)) if ref.TRACES else 0.0
    return out
