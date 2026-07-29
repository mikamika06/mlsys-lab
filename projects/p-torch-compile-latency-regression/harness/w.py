import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

BATCHES = [1, 2, 3, 4, 5, 6, 7, 8]


def torch_or_none():
    try:
        import torch
        return torch
    except ImportError:
        return None


def baseline():
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    mod = load("mlsys_baseline_model", os.path.join(HERE, "baseline_model.py"))
    pre = load("mlsys_baseline_preprocess", os.path.join(HERE, "baseline_preprocess.py"))
    return mod.Classifier, pre.normalise


def learner_service(workdir):
    from service.model import Classifier
    from service.preprocess import normalise
    return Classifier, normalise


def needs_torch(out):
    out["_note"] = "this milestone needs torch: pip install torch"
    return out
