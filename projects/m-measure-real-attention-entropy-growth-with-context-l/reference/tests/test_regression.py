import sys
import numpy as np

sys.path.insert(0, ".")
from longctx_eval.diagnostics import diagnose_models

def test_dilution_classification():
    accuracies = np.array([[1.0, 1.0, 0.2]])
    max_entropies = np.array([[2.0, 4.0, 15.0]])
    lengths = np.array([1000, 10000, 100000])

    res = diagnose_models(accuracies, max_entropies, lengths, 0.5, 10.0)
    assert res[0]['mode'] == 'dilution'

def test_rope_classification():
    accuracies = np.array([[1.0, 1.0, 0.2]])
    max_entropies = np.array([[2.0, 4.0, 5.0]])
    lengths = np.array([1000, 10000, 100000])

    res = diagnose_models(accuracies, max_entropies, lengths, 0.5, 10.0)
    assert res[0]['mode'] == 'rope'
