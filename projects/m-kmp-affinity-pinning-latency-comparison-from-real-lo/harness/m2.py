import ref

def check(workdir):
    from affinity.classify import classify_subscription
    correct = 0
    for cfg in ref.CONFIGS:
        want = ref.classify_state(cfg)
        got = classify_subscription(cfg)
        if got == want:
            correct += 1
    acc = float(correct) / float(len(ref.CONFIGS))
    return {"classification_accuracy": acc}
