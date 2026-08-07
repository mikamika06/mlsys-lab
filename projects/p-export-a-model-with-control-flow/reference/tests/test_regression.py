import sys
sys.path.insert(0, ".")
from exporter.model import business_logic_model
from exporter.core import export_model
import numpy as np

def test_export_contract():
    x = np.array([0.1, 0.8, 0.4, 0.9])
    res = business_logic_model(x, 4)
    assert len(res) == 4
    exported = export_model(business_logic_model, (x, 4))
    assert exported["status"] == "success"
