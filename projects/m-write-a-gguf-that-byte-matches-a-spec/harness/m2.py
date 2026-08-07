import ref

def check(workdir):
    from gguf_spec.classifier import classify_gguf
    fixtures = ref.get_corrupted_fixtures()
    correct = 0
    for i, (data, expected) in enumerate(fixtures):
        try:
            res = classify_gguf(data)
            if res == expected:
                correct += 1
        except Exception:
            pass
    out = {"classified_correctly": float(correct)}
    if correct < len(fixtures):
        out["_note"] = f"classified {correct}/{len(fixtures)} correctly"
    return out
