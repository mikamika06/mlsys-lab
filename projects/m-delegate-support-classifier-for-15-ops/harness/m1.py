import ref


def check(workdir):
    from delpipeline import classifier

    graphs = ref.generate_test_graphs()
    ok = 0
    total = len(graphs)
    for i, g in enumerate(graphs):
        want = classifier.classify_support(g)
        ref_out = [op["supported"] for op in g["ops"]]
        if want == ref_out:
            ok += 1

    score = float(ok) / float(total)
    out = {"classifier_match": score}
    if score < 1.0:
        out["_note"] = f"Classifier failed on {total - ok} graphs."
    return out
