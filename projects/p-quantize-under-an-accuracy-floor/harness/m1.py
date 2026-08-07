import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from quant.eval import Evaluator

    out = {"eval_accuracy_matches_ref": 0.0, "eval_runs_cleanly": 0.0}

    model = ref.MockModel()
    dataset = ref.get_dataset()

    try:
        evaluator = Evaluator(dataset)
        acc = evaluator.evaluate(model)
        out["eval_runs_cleanly"] = 1.0
    except Exception:
        return out

    correct = 0
    for x, target in dataset:
        pred = int(np.argmax(model.forward(x)))
        if pred == int(target):
            correct += 1
    expected_acc = correct / len(dataset)

    if abs(acc - expected_acc) < 1e-6:
        out["eval_accuracy_matches_ref"] = 1.0

    return out
