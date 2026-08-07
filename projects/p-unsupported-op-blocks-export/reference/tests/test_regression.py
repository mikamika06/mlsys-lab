import sys
import numpy as np

sys.path.insert(0, ".")
from exporter.replacements import catalog_add, substitute_op, check_tolerance

def test_replacement_catalog():
    res = catalog_add("CustomGelu", substitute_op)
    assert res is True

def test_export_pipeline():
    x = np.linspace(-2, 2, 40)
    y1 = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    y2 = substitute_op(x)
    assert check_tolerance(y1, y2, 1e-5) is True
